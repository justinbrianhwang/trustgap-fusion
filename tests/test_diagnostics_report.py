from pathlib import Path

from rrf.tools import diagnostics_report
from rrf.tools.diagnostics_report import (_chain_rows, _discover, _gate_rows, _partial_rows,
                                          _report, _trace_rows)


def test_diagnostics_report_reads_both_models_and_constant_q():
    entries = _discover([Path("tests/fixtures/diagnostics_report")])
    chain = _chain_rows(entries, repeats=20)
    gate = _gate_rows(entries)
    trace = _trace_rows(entries)

    assert len(entries) == 2 and len(trace) == 3
    b0 = next(row for row in chain if row["method"] == "B0")
    learned = next(row for row in chain if row["method"] == "P")
    assert b0["q_neg_E"] == b0["q_U"] == ""
    assert learned["q_neg_E"] < -0.9 and learned["q_U"] < -0.9
    tf_macro = next(row for row in gate if row["model"] == "TransFuser"
                    and row["condition_kind"] == "macro")
    assert tf_macro["cross_fitted_oracle_ap"] == 0.3
    assert tf_macro["learned_ap"] == 0.31 and tf_macro["temporal_oracle"] == 0.35
    report = _report(chain, gate, trace)
    assert report.count("# ") == 3 and "temporal_oracle" in report
    assert "TransFuser" in report and "InterFuser" in report


def test_diagnostics_report_builds_partial_table(monkeypatch):
    rows = []
    for index, q in enumerate((0.1, 0.3, 0.7, 0.9)):
        for modality in ("camera", "lidar"):
            for factor in (0.5, 0.8, 1.25, 2.0):
                delta = (q - 0.5) * (1 if factor > 1 else -1)
                rows.append({"sample_id": f"s{index}", "episode_id": f"ep{index // 2}",
                             "fg_present": "true", "kind": "seen", "modality": modality,
                             "factor": str(factor), "q": str(q), "E": str(1 - q),
                             "delta_loss": str(delta)})

    class FakePath:
        def with_name(self, _name):
            return self

        def exists(self):
            return True

        def read_text(self, **_kwargs):
            return '{"method":"P","seed":0}'

    monkeypatch.setattr(diagnostics_report, "_read_csv", lambda _path: rows)
    output = _partial_rows([("InterFuser", FakePath(), FakePath())], repeats=20)
    fraction = next(row for row in output if row["modality"] == "camera"
                    and row["statistic"] == "fraction_lower_f0.5")
    assert fraction["estimate"] == 0.5 and fraction["n_episodes"] == 2
