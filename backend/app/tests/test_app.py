def test_research_mode_notice():
    from app import research_mode_notice

    assert research_mode_notice("resumed") == (
        "This brief was produced from saved research data, "
        "not from a new live search."
    )
    assert research_mode_notice("live") == ""
