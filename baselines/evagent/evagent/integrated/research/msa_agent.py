"""Bounded evidence-building MSA agent. No prediction, labels, or provider calls.

Precomputed mode never invokes tools. Build mode searches reviewed family CMs,
retrieves natural homologs from a pinned local database, and aligns them with WT.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from pathlib import Path

from ..bundles import file_hash
from ..evidence_tools import canonical, empty_source, msa_features
from ..gateway import digest


def validate_usage(usage, prompt_sources=()):
    if usage not in {"external_fusion", "prompt_only"}:
        raise ValueError("Explicit MSA use must be external_fusion or prompt_only")
    if usage == "external_fusion" and "msa" in prompt_sources:
        raise ValueError("MSA cannot enter both the CR prompt and external fusion")


def pinned(record, base, *, natural=False):
    path = (base / record["path"]).resolve()
    if file_hash(path) != record.get("sha256"):
        raise ValueError("MSA asset hash mismatch")
    if record.get("reviewed") is not True or not record.get("source"):
        raise ValueError("MSA assets require reviewed source provenance")
    if natural and record.get("origin") != "natural_homologs":
        raise ValueError("Candidate libraries are not natural homolog databases")
    return path


def aligned_records(path, fmt):
    """Preserve WT insertions in rectangular A2M/Stockholm; reject ragged A3M.

    Automatic lower-case deletion loses WT coordinates. A3M callers must first
    expand insertion columns with an alignment-aware converter.
    """
    if fmt not in {"stockholm", "aligned_fasta", "a2m"}:
        raise ValueError("Use Stockholm or rectangular aligned_fasta/a2m, not ragged A3M")
    rows = {}
    name = None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        if fmt == "stockholm":
            if line.startswith(("#", "//")):
                continue
            name, sequence = line.split()
            rows[name] = rows.get(name, "") + sequence
        elif line.startswith(">"):
            name = line[1:].split()[0]
            if name in rows:
                raise ValueError("Duplicate alignment ID")
            rows[name] = ""
        else:
            if name is None:
                raise ValueError("Missing FASTA identifier")
            rows[name] += line.strip()
    rows = {n: s.upper().replace("T", "U").replace(".", "-") for n, s in rows.items()}
    if not rows or len({len(s) for s in rows.values()}) != 1:
        raise ValueError("Empty/ragged alignment; expand insertion columns first")
    if any(set(s) - set("ACGU-") for s in rows.values()):
        raise ValueError("Ambiguous alignment bases require explicit preprocessing")
    return rows


def features(wt, variants, path, fmt, query_id, output):
    rows = aligned_records(path, fmt)
    if query_id not in rows or rows[query_id].replace("-", "") != wt:
        raise ValueError("Named MSA query must match complete WT exactly")
    query = rows[query_id]
    # Query aliases cannot inflate depth. Homolog duplicate rows are also removed.
    homologs = sorted({s for name, s in rows.items()
                       if name != query_id and s.replace("-", "") != wt})
    normalized = output / "normalized.sto"
    normalized.write_text("# STOCKHOLM 1.0\nWT_query " + query + "\n" +
                          "".join(f"h{i} {s}\n" for i, s in enumerate(homologs)) + "//\n")
    if len(homologs) < 2:
        return empty_source("Fewer than two distinct non-query natural homologs")
    result = msa_features(wt, variants, normalized,
                          {"origin": "natural_homologs", "reviewed": True})
    columns = [i + 1 for i, base in enumerate(query) if base != "-"]
    # JSON object keys are strings. Normalize before hashing so coordinates 10+
    # retain the same canonical order when receipts and snapshots are reloaded.
    result["task"]["wt_to_alignment_column"] = {str(i): col for i, col in enumerate(columns, 1)}
    result["task"]["query_aliases_excluded"] = sum(s.replace("-", "") == wt for s in rows.values())
    return result


def search_hits(path, maximum_evalue):
    hits = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 17:
            raise ValueError("Malformed Infernal default tblout")
        evalue = float(fields[15])
        if not math.isfinite(evalue):
            raise ValueError("Nonfinite Infernal E-value")
        if evalue <= maximum_evalue:
            hits.append({"id": fields[0], "start": int(fields[7]), "end": int(fields[8]),
                         "strand": fields[9], "evalue": evalue})
    return sorted(hits, key=lambda h: (h["evalue"], h["id"], h["start"], h["end"]))


def extract_homologs(database, hits):
    wanted = {h["id"] for h in hits}
    records, chunks, name = {}, [], None

    def keep():
        if name in wanted:
            if name in records:
                raise ValueError("Duplicate natural database sequence ID")
            records[name] = "".join(chunks).upper().replace("T", "U")

    with database.open() as stream:
        for line in stream:
            if line.startswith(">"):
                keep()
                name, chunks = line[1:].split()[0], []
            elif name in wanted:
                chunks.append(line.strip())
        keep()
    output = []
    for hit in hits:
        sequence = records[hit["id"]]
        lo, hi = sorted((hit["start"], hit["end"]))
        if not 1 <= lo <= hi <= len(sequence) or hit["strand"] not in {"+", "-"}:
            raise ValueError("Invalid homolog coordinates")
        part = sequence[lo - 1:hi]
        if hit["strand"] == "-":
            part = part.translate(str.maketrans("ACGU", "UGCA"))[::-1]
        if set(part) <= set("ACGU"):
            output.append((f"homolog_{len(output)}", part, hit))
    return output


class MSAPreparationAgent:
    """Deterministic bounded tool policy; no LLM is needed to execute this workflow."""

    def __init__(self, config, base, output, runner=subprocess.run):
        self.config, self.base, self.output, self.runner = config, Path(base), Path(output), runner
        self.calls = []

    def run_tool(self, argv, name):
        maximum = self.config.get("max_tool_calls", 12)
        if type(maximum) is not int or maximum < 1 or len(self.calls) >= maximum:
            raise ValueError("MSA tool-call allowance exhausted")
        record = {"command": argv, "status": "inflight", "name": name}
        self.calls.append(record)
        log = self.output / "tool_events.jsonl"

        def event():
            import os
            with log.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

        event()
        start = time.monotonic()
        try:
            with (self.output / (name + ".log")).open("w") as stream:
                self.runner(argv, check=True, stdout=stream, stderr=subprocess.STDOUT,
                            timeout=self.config.get("timeout_s", 600))
            record["status"] = "ok"
            record["output_log_sha256"] = file_hash(self.output / (name + ".log"))
        except BaseException as error:
            record.update(status="error", error_type=type(error).__name__)
            raise
        finally:
            record["elapsed_s"] = time.monotonic() - start
            event()

    def prepare(self, wt_sequence, variants=()):
        cfg = self.config
        validate_usage(cfg.get("usage", "external_fusion"), cfg.get("prompt_sources", []))
        wt = canonical(wt_sequence)
        self.output.mkdir(parents=True, exist_ok=False)
        mode = cfg.get("mode", "precomputed")
        if mode == "precomputed":
            asset = cfg.get("alignment")
            if asset is None:
                result = empty_source("Precomputed MSA absent; construction disabled")
            else:
                path = pinned(asset, self.base, natural=True)
                result = features(wt, list(variants), path, asset["format"],
                                  asset["query_identifier"], self.output)
        elif mode == "build":
            result = self.build(wt, list(variants))
        else:
            raise ValueError("Unknown MSA preparation mode")
        result["provenance"].append({"tool": "evagent-msa-preparation-agent", "version": "v2",
                                     "mode": mode, "config_sha256": digest(cfg),
                                     "wt_sha256": digest(wt), "sha256": digest(result)})
        receipt = {"schema": "evagent-msa-agent-v2", "usage": cfg.get("usage", "external_fusion"),
                   "config": cfg, "source": result, "tool_calls": self.calls}
        receipt["sha256"] = digest(receipt)
        (self.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return result

    def build(self, wt, variants):
        cfg = self.config
        families = cfg.get("families", [])
        maximum = cfg.get("max_families", 3)
        if type(maximum) is not int or maximum < 1 or len(families) > maximum:
            raise ValueError("Supply a bounded reviewed family search catalog")
        if families and cfg.get("family_library"):
            raise ValueError("Choose explicit families or automatic library discovery")
        if not families and not cfg.get("family_library"):
            raise ValueError("Supply family CMs or a pressed Rfam library")
        database = pinned(cfg["database"], self.base, natural=True)
        tools = {name: pinned(cfg["tools"][name], self.base) for name in ("cmsearch", "cmalign")}
        for name, executable in tools.items():
            self.run_tool([str(executable), "-h"], name + "_version")
        query = self.output / "query.fa"
        query.write_text(">WT_query\n" + wt + "\n")
        threshold = cfg.get("max_evalue", .001)
        if not isinstance(threshold, (int, float)) or not 0 < threshold <= 1:
            raise ValueError("Invalid E-value threshold")
        if cfg.get("family_library"):
            families = self.discover_families(query, maximum, threshold)
            if not families:
                return empty_source("No significant forward-strand family hit in pinned Rfam library")
        choices = []
        for i, family in enumerate(families):
            cm = pinned(family, self.base)
            if family.get("family_discovery"):
                choices.append((family["family_discovery"]["evalue"], i, cm))
                continue
            table = self.output / f"family_{i}.tbl"
            self.run_tool([str(tools["cmsearch"]), "--cpu", "1", "--noali", "--tblout",
                           str(table), "-E", str(threshold), str(cm), str(query)], f"family_{i}")
            hits = [h for h in search_hits(table, threshold) if h["id"] == "WT_query" and h["strand"] == "+"]
            if hits:
                choices.append((hits[0]["evalue"], i, cm))
        for _, index, cm in sorted(choices):
            table = self.output / f"homolog_{index}.tbl"
            self.run_tool([str(tools["cmsearch"]), "--cpu", "1", "--noali", "--tblout",
                           str(table), "-E", str(threshold), str(cm), str(database)], f"homolog_{index}")
            homologs = extract_homologs(database, search_hits(table, threshold)[:2048])
            if len({s for _, s, _ in homologs if s != wt}) < 2:
                continue
            fasta = self.output / f"homolog_{index}.fa"
            fasta.write_text(query.read_text() + "".join(f">{n}\n{s}\n" for n, s, _ in homologs))
            (self.output / f"homolog_{index}_coordinates.json").write_text(json.dumps(homologs))
            alignment = self.output / f"aligned_{index}.sto"
            self.run_tool([str(tools["cmalign"]), "--cpu", "1", "--outformat", "Stockholm",
                           "-o", str(alignment), str(cm), str(fasta)], f"align_{index}")
            result = features(wt, variants, alignment, "stockholm", "WT_query", self.output)
            result["task"]["family"] = families[index].get("accession", str(index))
            result["limitations"].append("Family search is limited to the configured catalog; no-hit does not prove absence of homologs.")
            return result
        return empty_source("No reviewed family produced sufficient unambiguous natural homologs")

    def discover_families(self, query, maximum, threshold):
        """Search the complete pinned Rfam library before bounded family trials."""
        library = self.config["family_library"]
        original = pinned(library, self.base)
        local = self.output / "Rfam.cm"
        shutil.copy2(original, local)
        for suffix in (".i1f", ".i1i", ".i1m", ".i1p"):
            index = pinned(library["indices"][suffix], self.base)
            shutil.copy2(index, Path(str(local) + suffix))
        scan = pinned(self.config["tools"]["cmscan"], self.base)
        fetch = pinned(self.config["tools"]["cmfetch"], self.base)
        for name, executable in (("cmscan", scan), ("cmfetch", fetch)):
            self.run_tool([str(executable), "-h"], name + "_version")
        table = self.output / "discovery.tbl"
        self.run_tool([str(scan), "--cpu", "1", "--noali", "--tblout", str(table),
                       "-E", str(threshold), str(local), str(query)], "discovery")
        hits = [h for h in search_hits(table, threshold) if h["strand"] == "+"]
        families, seen = [], set()
        for hit in hits:
            if hit["id"] in seen:
                continue
            seen.add(hit["id"])
            destination = self.output / f"discovered_{len(families)}.cm"
            self.run_tool([str(fetch), "-o", str(destination), str(local), hit["id"]],
                          f"fetch_{len(families)}")
            families.append({"path": str(destination), "sha256": file_hash(destination),
                             "source": library["source"], "reviewed": True,
                             "accession": hit["id"], "family_discovery": hit})
            if len(families) >= maximum:
                break
        # The original pinned library remains untouched; all scratch is local.
        return families
