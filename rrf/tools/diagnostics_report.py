"""Combine InterFuser and TransFuser diagnostic outputs into three compact tables."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from rrf.diag_common import write_csv
from rrf.stats import episode_bootstrap_ci


GATE_FIELDS = {
    "learned_ap": ("utility", "ap", "full"),
    "neutral_ap": ("oracle_gate", "neutral_ap"),
    "reliability_oracle_absolute_ap": ("oracle_rel", "ap", "absolute"),
    "reliability_oracle_relative_ap": ("oracle_rel", "ap", "relative"),
    "best_fixed_ap": ("oracle_gate", "best_fixed_ap"),
    "in_sample_oracle_ap": ("oracle_gate", "oracle_ap"),
    "cross_fitted_oracle_ap": ("oracle_gate_xfit", "ap", "cross_fitted_oracle"),
    "temporal_oracle": ("oracle_gate_temporal", "ap", "temporal_oracle"),
    "global_shuffle_ap": ("shuffle", "ap", "global"),
    "within_condition_shuffle_ap": ("shuffle", "ap", "within_condition"),
}


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rank(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(rows, left, right):
    x = np.asarray([float(row[left]) for row in rows])
    y = np.asarray([float(row[right]) for row in rows])
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def _ci_fields(rows, left, right, repeats, seed):
    result = episode_bootstrap_ci(
        rows, lambda sample: _spearman(sample, left, right), n=repeats, seed=seed)
    return result["estimate"], result["ci_low"], result["ci_high"]


def _number(value):
    return "" if value is None or not np.isfinite(value) else float(value)


def _discover(roots):
    found = []
    for root in map(Path, roots):
        directories = [(root / "diagnostics", "InterFuser"),
                       (root / "tf_diagnostics", "TransFuser")]
        if root.name == "diagnostics":
            directories = [(root, "InterFuser")]
        elif root.name == "tf_diagnostics":
            directories = [(root, "TransFuser")]
        for diagnostics, model in directories:
            if not diagnostics.is_dir():
                continue
            for summary_path in sorted(diagnostics.glob("*/summary.json")):
                utility_path = summary_path.with_name("utility.csv")
                if utility_path.exists():
                    found.append((model, summary_path, utility_path))
    return found


def _identity(model, summary):
    return {"model": model, "method": summary.get("method", ""),
            "seed": summary.get("seed", ""), "variant": summary.get("variant", "")}


def _chain_rows(entries, repeats):
    output = []
    for model, summary_path, utility_path in entries:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        identity = _identity(model, summary)
        utility = _read_csv(utility_path)
        foreground = [row for row in utility if row.get("fg_present", "").lower() == "true"]
        groups = sorted({(row["variant"], row["modality"]) for row in foreground})
        for removal_variant, modality in groups:
            rows = [row for row in foreground
                    if row["variant"] == removal_variant and row["modality"] == modality]
            q_constant = len({row["q"] for row in rows}) < 2
            values = {}
            for prefix, left, right in (("q_neg_E", "q", "neg_E"),
                                        ("q_U", "q", "utility"),
                                        ("neg_E_U", "neg_E", "utility")):
                if q_constant and left == "q":
                    estimate = low = high = None
                else:
                    estimate, low, high = _ci_fields(
                        rows, left, right, repeats, int(summary.get("seed", 0)))
                values.update({prefix: _number(estimate), f"{prefix}_ci_low": _number(low),
                               f"{prefix}_ci_high": _number(high)})
            output.append(dict(identity, modality=modality, removal_variant=removal_variant,
                               n_frames=len(rows), **values))
    return output


def _at(mapping, path):
    for key in path:
        if not isinstance(mapping, dict) or key not in mapping:
            return {}
        mapping = mapping[key]
    return mapping if isinstance(mapping, dict) else {}


def _ap(value):
    if isinstance(value, dict):
        value = value.get("heatmap_peak_ap")
    return None if value is None else float(value)


def _mean(values):
    values = [value for value in values if value is not None]
    return None if not values else float(np.mean(values))


def _gate_rows(entries):
    output = []
    for model, summary_path, utility_path in entries:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        analyses = summary.get("analyses", {})
        identity = _identity(model, summary)
        utility = _read_csv(utility_path)
        condition_kind = {row["condition"]: row["kind"] for row in utility}
        xfit = analyses.get("oracle_gate_xfit", {})
        temporal = analyses.get("oracle_gate_temporal", {})
        sources = {name: _at(analyses, path) for name, path in GATE_FIELDS.items()}
        for name in ("learned_ap", "neutral_ap", "best_fixed_ap", "in_sample_oracle_ap"):
            xfit_name = {"learned_ap": "learned", "neutral_ap": "neutral",
                         "best_fixed_ap": "best_fixed",
                         "in_sample_oracle_ap": "in_sample_oracle"}[name]
            if _at(xfit, ("ap", xfit_name)):
                sources[name] = _at(xfit, ("ap", xfit_name))
        for name, temporal_name in (("learned_ap", "learned"), ("neutral_ap", "neutral"),
                                    ("best_fixed_ap", "best_fixed"),
                                    ("in_sample_oracle_ap", "in_sample_oracle"),
                                    ("temporal_oracle", "temporal_oracle")):
            if _at(temporal, ("ap", temporal_name)):
                sources[name] = _at(temporal, ("ap", temporal_name))
        conditions = sorted(set(condition_kind) | {condition for source in sources.values()
                                                    for condition in source})
        kind_rows = []
        for kind in sorted(set(condition_kind.values())):
            members = [condition for condition in conditions if condition_kind.get(condition) == kind]
            values = {name: _number(_mean([_ap(source.get(condition))
                                           for condition in members]))
                      for name, source in sources.items()}
            kind_rows.append(dict(identity, condition_kind=kind, n_conditions=len(members), **values))
        output.extend(kind_rows)
        corruption_rows = [row for row in kind_rows if row["condition_kind"] != "clean"]
        values = {name: _number(_mean([None if row[name] == "" else row[name]
                                       for row in corruption_rows])) for name in GATE_FIELDS}
        output.append(dict(identity, condition_kind="macro", n_conditions=sum(
            row["n_conditions"] for row in corruption_rows), **values))
    return output


def _trace_rows(entries):
    totals = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for model, summary_path, _ in entries:
        trace_path = summary_path.with_name("trace.csv")
        if not trace_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        identity = _identity(model, summary)
        for row in _read_csv(trace_path):
            swept_modality, q_value = row["variant"].rsplit("_", 1)
            masses = []
            if "encoder_camera_mass" in row:
                for stage in ("encoder", "decoder"):
                    camera = json.loads(row[f"{stage}_camera_mass"])
                    lidar = json.loads(row[f"{stage}_lidar_mass"])
                    masses.extend((stage, layer, c, l) for layer, (c, l) in enumerate(zip(camera, lidar)))
            else:
                for field, value in row.items():
                    if not field.endswith("_camera_mass"):
                        continue
                    prefix = field[:-len("_camera_mass")]
                    stage, layer = prefix.rsplit("_layer_", 1)
                    masses.append((stage, int(layer), float(value), float(row[f"{prefix}_lidar_mass"])))
            for stage, layer, camera, lidar in masses:
                key = (*identity.values(), stage, layer, swept_modality, float(q_value))
                total = totals[key]
                total[0] += float(camera)
                total[1] += float(lidar)
                total[2] += float(row["mean_abs_delta_score"])
                total[3] += 1
    output = []
    for key, (camera, lidar, delta, count) in sorted(totals.items()):
        model, method, seed, variant, stage, layer, swept_modality, q_value = key
        output.append({"model": model, "method": method, "seed": seed, "variant": variant,
                       "stage": stage, "layer": layer, "swept_modality": swept_modality,
                       "q_value": q_value, "camera_attention_mass": camera / count,
                       "lidar_attention_mass": lidar / count,
                       "mean_abs_delta_score": delta / count, "n_samples": count})
    return output


def _fmt(value):
    if value == "" or value is None:
        return ""
    if isinstance(value, str):
        return value
    return f"{float(value):.3f}"


def _correlation(row, prefix):
    if row[prefix] == "":
        return ""
    return f"{_fmt(row[prefix])} [{_fmt(row[prefix + '_ci_low'])}, {_fmt(row[prefix + '_ci_high'])}]"


def _markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return lines


def _report(chain, gate, trace):
    lines = ["# Chain links", ""]
    lines += _markdown_table(
        ["Model", "Method", "Seed", "Removal", "Modality", "ρ(q,−E)", "ρ(q,U)", "ρ(−E,U)"],
        [[row["model"], row["method"], row["seed"], row["removal_variant"], row["modality"],
          _correlation(row, "q_neg_E"), _correlation(row, "q_U"),
          _correlation(row, "neg_E_U")] for row in chain])
    lines += ["", "# Gate effect", ""]
    macro = [row for row in gate if row["condition_kind"] == "macro"]
    gate_columns = list(GATE_FIELDS)
    lines += _markdown_table(
        ["Model", "Method", "Seed", *[name.removesuffix("_ap") for name in gate_columns]],
        [[row["model"], row["method"], row["seed"], *[_fmt(row[name]) for name in gate_columns]]
         for row in macro])
    lines += ["", "# Trace", ""]
    grouped = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for row in trace:
        key = (row["model"], row["method"], row["seed"], row["swept_modality"], row["q_value"])
        total = grouped[key]
        for index, field in enumerate(("camera_attention_mass", "lidar_attention_mass",
                                       "mean_abs_delta_score")):
            total[index] += row[field]
        total[3] += 1
    trace_macro = [(*key, *(value / total[3] for value in total[:3]))
                   for key, total in sorted(grouped.items())]
    lines += _markdown_table(
        ["Model", "Method", "Seed", "Swept", "q", "Camera mass", "LiDAR mass", "Mean abs Δscore"],
        [[*row[:4], _fmt(row[4]), _fmt(row[5]), _fmt(row[6]), _fmt(row[7])]
         for row in trace_macro])
    return "\n".join(lines) + "\n"


def build_report(roots, out, repeats=2000):
    entries = _discover(roots)
    if not entries:
        raise RuntimeError("no diagnostic summary.json + utility.csv pairs found")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    chain, gate, trace = _chain_rows(entries, repeats), _gate_rows(entries), _trace_rows(entries)
    write_csv(out / "chain_links.csv", chain)
    write_csv(out / "gate_effect.csv", gate)
    if trace:
        write_csv(out / "trace.csv", trace)
    else:
        (out / "trace.csv").write_text(
            "model,method,seed,variant,stage,layer,swept_modality,q_value,camera_attention_mass,"
            "lidar_attention_mass,mean_abs_delta_score,n_samples\n", encoding="utf-8")
    (out / "report.md").write_text(_report(chain, gate, trace), encoding="utf-8")
    return {"diagnostic_runs": len(entries), "chain_links": len(chain),
            "gate_effect": len(gate), "trace": len(trace)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_repeats < 1:
        parser.error("--bootstrap-repeats must be >= 1")
    print(json.dumps(build_report(args.roots, args.out, args.bootstrap_repeats), indent=2))


if __name__ == "__main__":
    main()
