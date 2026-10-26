"""Read-only audit of provider-returned summaries; no inference/API calls.

This postprocessing script is outside the frozen predictor package. It never
loads oracle/test labels and never edits requests, caches, or budget ledgers.
"""
import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

PATTERNS = {
    "measured_examples": r"\b(?:S\d{3}|measured|experimental|support|examples?|few.shot|references?)\b",
    "scale_calibration": r"\b(?:median|IQR|atan|calibrat\w*|raw (?:value|score|readout)|scale)\b",
    "nearby_variants": r"\b(?:similar|nearest|neighbor|neighbour|analog\w*|interpolat\w*)\b",
    "mutation_interactions": r"\b(?:epistas\w*|interaction|combin\w*|compensat\w*)\b",
    "biophysical_prior": r"\b(?:stability|folding|cataly\w*|binding|base.pair|stem|hydrophob\w*)\b",
    "uncertainty": r"\b(?:uncertain\w*|ambiguous|confidence|speculat\w*|cannot determine)\b",
}


def audit(root):
    groups = defaultdict(Counter)
    examples = {}
    cells = []
    for path in sorted((root / "predictions").glob("*/result.json")):
        result = json.loads(path.read_text())
        request = json.loads(path.with_name("request.json").read_text())
        summaries = [a.get("thought_summary", "") for a in result.get("attempts", [])]
        text = "\n".join(s for s in summaries if isinstance(s, str))
        group = request.get("arm", request.get("condition", {}).get("name", "unknown"))
        counts = groups[group]
        counts["requests"] += 1
        counts["with_returned_summary"] += bool(text.strip())
        flags = {name: bool(re.search(pattern, text, re.I)) for name, pattern in PATTERNS.items()}
        for name, value in flags.items():
            counts[name] += value
        # Strictly a visible-output audit, not evidence of causal label use.
        cells.append({"request_hash": result["request_hash"], "assay": request["episode_id"].split("/")[0],
                      "arm": group, "status": result["status"], "summary_chars": len(text), **flags})
        if text and group not in examples:
            sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
            selected = [s for s in sentences if re.search(PATTERNS["measured_examples"], s, re.I)]
            excerpt = " ".join((selected or sentences)[:2])[:600].replace("|", "/")
            examples[group] = {"request_hash": result["request_hash"], "excerpt": excerpt}
    output = root / "analysis"
    output.mkdir(exist_ok=True)
    data = {"snapshot_epoch": time.time(), "groups": {k: dict(v) for k, v in groups.items()},
            "request_audit": cells, "examples": examples,
            "limitations": ["Only provider-returned summaries, not complete internal reasoning.",
                            "Keyword mentions do not establish causal label use or correctness.",
                            "No additional model calls; assay results remain in the independent paired report."]}
    temporary = output / "thought_audit.json.tmp"
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(output / "thought_audit.json")
    lines = ["# Observable thought-summary audit", "", f"Snapshot epoch: {data['snapshot_epoch']:.0f}", "",
             "This is a reproducible keyword audit of returned summaries. Mentions of support labels or a calibration formula do not prove that labels causally improved the predictions. These are not full internal reasoning traces.", "",
             "|Arm|Completed requests|Returned summaries|Measured/example mentions|Scale|Neighbors|Interactions|Biophysics|Uncertainty|",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, count in sorted(groups.items()):
        lines.append("|" + "|".join([arm, str(count["requests"]), str(count["with_returned_summary"])] +
                                    [str(count[k]) for k in PATTERNS]) + "|")
    lines += ["", "## Illustrative excerpts", "", "First available request per arm in sorted hash order; not selected by fitness or model performance."]
    for arm, example in sorted(examples.items()):
        lines += ["", f"### {arm}", "", f"Request `{example['request_hash']}`.", "", "> " + example["excerpt"].replace("\n", " ")]
    temporary = output / "THOUGHT_SUMMARIES.md.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(output / "THOUGHT_SUMMARIES.md")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    audit(parser.parse_args().root)
