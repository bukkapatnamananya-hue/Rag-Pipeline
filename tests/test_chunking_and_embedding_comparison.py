from app.chunking.comparison import build_strategy_comparison


def test_chunking_strategies_return_expected_profiles():
    sample_text = (
        "Zen Technologies Limited is a leading defense technology company. "
        "The company develops anti-drone systems and training simulators. "
        "Revenue increased materially during FY24 while EBITDA and PAT improved. "
        "The order book remains strong with defense and training programs."
    )

    report = build_strategy_comparison("demo-doc", [sample_text])

    assert set(report.keys()) >= {"fixed_size", "recursive", "contextual"}
    for strategy, payload in report.items():
        assert payload["count"] > 0
        assert payload["avg_chunk_length"] > 0
        assert len(payload["chunks"]) == payload["count"]
