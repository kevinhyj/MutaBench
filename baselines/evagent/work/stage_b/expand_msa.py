"""Search admitted WT inputs against the full frozen Rfam CM release.

Only biologically reviewed family matches can supply evidence. All search
receipts are retained, including no-hits; candidate variants are never queried.
Existing structures and their recorded MSA dependencies are left untouched.
"""

import argparse
import gzip
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import httpx

from evagent.integrated.bundles import file_hash
from evagent.integrated.evidence_tools import canonical, read_alignment

TOOLS = Path(
    "/data/yanjie_huang/EVAgent/work/stage_b_tools/infernal-1.1.5-linux-intel-gcc/binaries"
)
COMPATIBLE = {
    "yeast_tRNA": {"RF00005"},
    "mammalian_CPEB3": {"RF00622"},
    "Azoarcus_groupI": {"RF00028"},
    "rice_twister": {"RF03160", "RF03154", "RF02684"},
    "human_TS_like": {"RF02681"},
}


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def fetch(client, url, path):
    if not path.exists():
        response = client.get(url)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    root = parser.parse_args().assets.resolve()
    public = root / "public/rfam-expanded"
    public.mkdir(parents=True, exist_ok=True)
    registry = json.loads((root / "registry.json").read_text())
    admitted = [r for r in registry["ncrna_tasks"] if r["audit"]["status"] == "admitted"]
    tasks = {
        r["assay"]: json.loads((root / "tasks" / f"{r['assay']}.json").read_text())
        for r in admitted
    }
    query = public / "admitted_wt.fa"
    query.write_text("".join(f">{a}\n{canonical(t['wt_sequence'])}\n" for a, t in tasks.items()))
    receipt = {
        "query_scope": "admitted complete WT only; no mutant sequences or scores",
        "query_sha256": file_hash(query),
        "infernal_version": "1.1.5",
        "families": {},
        "tasks": {},
    }
    with httpx.Client(timeout=300, follow_redirects=True, trust_env=False) as client:
        archive = fetch(
            client,
            "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.cm.gz",
            public / "Rfam.cm.gz",
        )
        cm = public / "Rfam.cm"
        if not cm.exists():
            cm.write_bytes(gzip.decompress(archive.read_bytes()))
        text = cm.read_text()
        families = {}
        for block in text.split("//"):
            fields = {
                p[0]: p[1] for line in block.splitlines() if len(p := line.split(maxsplit=1)) == 2
            }
            if "ACC" in fields and "NAME" in fields:
                families[fields["ACC"]] = fields["NAME"]
        receipt.update(
            cm_archive_sha256=file_hash(archive),
            cm_sha256=file_hash(cm),
            family_count=len(families),
        )
        save(public / "family_catalog.json", families)
        print(json.dumps({"stage": "CM downloaded", "families": len(families)}), flush=True)
        if not Path(str(cm) + ".i1f").exists():
            subprocess.run([str(TOOLS / "cmpress"), str(cm)], check=True, stdout=subprocess.DEVNULL)
        table = public / "cmscan.tbl"
        args = [
            str(TOOLS / "cmscan"),
            "--cpu",
            "2",
            "--rfam",
            "--cut_ga",
            "--noali",
            "--tblout",
            str(table),
            str(cm),
            str(query),
        ]
        if not (public / "cmscan.done.json").exists():
            with (public / "cmscan.log").open("w") as log:
                subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
            save(public / "cmscan.done.json", {"command": args, "table_sha256": file_hash(table)})
        receipt["search_command"] = args
        hits = {a: [] for a in tasks}
        for line in table.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            s = line.split()
            hits[s[2]].append(
                {
                    "name": s[0],
                    "accession": s[1],
                    "start": int(s[7]),
                    "end": int(s[8]),
                    "strand": s[9],
                    "score": float(s[14]),
                    "evalue": float(s[15]),
                    "inclusion": s[16],
                }
            )
        # Audit every official family/assay alignment instead of only *.a2m.
        z = zipfile.ZipFile(root / "public/fitness_MSAs.zip")
        receipt["official_archive_sha256"] = file_hash(root / "public/fitness_MSAs.zip")
        receipt["official_alignment_inventory"] = [
            {"path": n, "records": z.read(n).count(b">")}
            for n in z.namelist()
            if n.endswith((".fa", ".a2m", ".fasta"))
        ]
        # A sensitive second pass for independently identified families removes
        # heuristic filters, with thresholds fixed before observing its hits.
        targeted = {}
        for family in ("Azoarcus_groupI", "rice_twister", "human_TS_like"):
            members = [r["assay"] for r in admitted if r["audit"]["family"] == family]
            qpath = public / f"{family}.fa"
            qpath.write_text(
                "".join(f">{a}\n{canonical(tasks[a]['wt_sequence'])}\n" for a in members)
            )
            for accession in sorted(COMPATIBLE[family]):
                famdir = public / accession
                famdir.mkdir(exist_ok=True)
                fcm = fetch(client, f"https://rfam.org/family/{accession}/cm", famdir / "family.cm")
                tab = famdir / "targeted.tbl"
                cmd = [
                    str(TOOLS / "cmsearch"),
                    "--cpu",
                    "2",
                    "--max",
                    "--toponly",
                    "-E",
                    "0.001",
                    "--noali",
                    "--tblout",
                    str(tab),
                    str(fcm),
                    str(qpath),
                ]
                if not tab.exists():
                    with (famdir / "targeted.log").open("w") as log:
                        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
                for line in tab.read_text().splitlines():
                    if line.startswith("#") or not line.strip():
                        continue
                    s = line.split()
                    a = s[0]
                    h = {
                        "name": s[2],
                        "accession": s[3],
                        "start": int(s[7]),
                        "end": int(s[8]),
                        "strand": s[9],
                        "score": float(s[14]),
                        "evalue": float(s[15]),
                        "inclusion": s[16],
                        "search": "targeted_no_filters",
                        "acceptance": "independent family identity, E<=0.001, score>=20 bits, >=50pct WT coverage",
                    }
                    targeted.setdefault(a, []).append(h)
                    if h["score"] >= 20 and abs(h["end"] - h["start"]) + 1 >= 0.5 * len(
                        canonical(tasks[a]["wt_sequence"])
                    ):
                        hits[a].append(h)
        receipt["targeted_pass"] = {
            "thresholds": {"max_evalue": 0.001, "min_bits": 20, "min_wt_coverage": 0.5},
            "hits": targeted,
        }
        for row in admitted:
            assay = row["assay"]
            family = row["audit"]["family"]
            wt = canonical(tasks[assay]["wt_sequence"])
            folder = root / "evidence" / assay
            raw = folder / "raw"
            compatible = COMPATIBLE.get(family, set())
            accepted = [
                h
                for h in hits[assay]
                if h["accession"] in compatible
                and h["strand"] == "+"
                and abs(h["end"] - h["start"]) + 1 >= 0.5 * len(wt)
            ]
            record = {
                "hits": hits[assay],
                "allowed_families": sorted(compatible),
                "status": "no_compatible_significant_hit",
            }
            # Keep previously verified mappings; attempt FULL CPEB3 homologs too.
            oldaudit = (
                json.loads((folder / "msa_audit.json").read_text())
                if (folder / "msa_audit.json").exists()
                else None
            )
            chosen = (
                max(accepted, key=lambda h: h["score"])["accession"]
                if accepted
                else (oldaudit or {}).get("family")
            )
            if not chosen:
                record["reason"] = (
                    "Full Rfam catalogue searched; no compatible hit passing either catalogue GA or predefined sensitive targeted thresholds. Synthetic selected families have no asserted natural homologs."
                )
                receipt["tasks"][assay] = record
                print(json.dumps({"assay": assay, **record}), flush=True)
                continue
            if oldaudit and family != "mammalian_CPEB3":
                record.update(
                    status="retained_verified_seed",
                    family=chosen,
                    audit_sha256=file_hash(folder / "msa_audit.json"),
                )
                receipt["tasks"][assay] = record
                continue
            famdir = public / chosen
            famdir.mkdir(exist_ok=True)
            cmurl = f"https://rfam.org/family/{chosen}/cm"
            seedurl = f"https://rfam.org/family/{chosen}/alignment"
            familycm = fetch(client, cmurl, famdir / "family.cm")
            seed = fetch(client, seedurl, famdir / "seed.sto")
            aligned = read_alignment(seed)
            homologs = [(n, s.replace("-", "")) for n, s in aligned if not set(s) - set("ACGU-")]
            dataset = "seed"
            fullurl = f"https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/fasta_files/{chosen}.fa.gz"
            if family in {"mammalian_CPEB3", "Azoarcus_groupI"}:
                try:
                    full = fetch(client, fullurl, famdir / "full.fa.gz")
                    content = gzip.decompress(full.read_bytes()).decode()
                    records = []
                    for block in content.split(">")[1:]:
                        name, *lines = block.splitlines()
                        seq = "".join(lines).upper().replace("T", "U")
                        if seq and not set(seq) - set("ACGU"):
                            records.append((name.split()[0], seq))
                    if len({s for _, s in records}) > len({s for _, s in homologs}):
                        homologs = records
                        dataset = "full"
                    record["full_download"] = {
                        "status": "ok",
                        "records": len(records),
                        "sha256": file_hash(full),
                        "url": fullurl,
                    }
                except (httpx.HTTPError, ValueError, OSError) as error:
                    record["full_download"] = {
                        "status": "unavailable",
                        "error": str(error),
                        "url": fullurl,
                    }
            unique = {}
            for n, s in sorted(homologs):
                unique.setdefault(s, n)
            homologs = [
                (unique[s], s)
                for s in sorted(
                    unique, key=lambda s: hashlib.sha256(("seed1:" + s).encode()).hexdigest()
                )
            ][:2048]
            fasta = raw / "expanded_natural.fa"
            fasta.write_text(
                ">WT_query\n"
                + wt
                + "\n"
                + "".join(f">natural_{i}\n{s}\n" for i, (_, s) in enumerate(homologs))
            )
            save(
                raw / "expanded_identifiers.json",
                {
                    "records": [
                        {"id": f"natural_{i}", "accession": n} for i, (n, _) in enumerate(homologs)
                    ]
                },
            )
            output = raw / "expanded_mapped.sto"
            cmd = [
                str(TOOLS / "cmalign"),
                "--cpu",
                "2",
                "--notrunc",
                "-o",
                str(output),
                str(familycm),
                str(fasta),
            ]
            with (raw / "expanded_cmalign.log").open("w") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            assert dict(read_alignment(output))["WT_query"].replace("-", "") == wt
            audit = {
                "origin": "natural_homologs",
                "reviewed": True,
                "family": chosen,
                "family_name": families[chosen],
                "query_identifier": "WT_query",
                "infernal_version": "1.1.5",
                "dataset": dataset,
                "selection": "unique sequences; lowest SHA256(seed1:sequence), cap 2048",
                "source_urls": [cmurl, seedurl] + ([fullurl] if dataset == "full" else []),
                "seed_sha256": file_hash(seed),
                "cm_sha256": file_hash(familycm),
                "natural_input_sha256": file_hash(fasta),
                "identifier_map_sha256": file_hash(raw / "expanded_identifiers.json"),
                "parameters": cmd[1:4],
                "family_match": accepted,
                "note": "Complete WT mapped to frozen natural homologs; synthetic mutant libraries excluded. No change to AF3 inputs or structure evidence.",
            }
            save(folder / "expanded_msa_audit.json", audit)
            record.update(
                status="expanded_alignment_ready",
                family=chosen,
                dataset=dataset,
                natural_records=len(homologs),
            )
            receipt["tasks"][assay] = record
            save(public / "search_receipt.json", receipt)
            print(json.dumps({"assay": assay, **record}), flush=True)
        save(public / "search_receipt.json", receipt)


if __name__ == "__main__":
    main()
