from pathlib import Path

from rrf.tools.diagnostics_report import (_chain_rows, _discover, _gate_rows,
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
