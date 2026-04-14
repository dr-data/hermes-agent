from agent.collective_intelligence_store import CollectiveIntelligenceStore


def test_collective_store_record_and_search(tmp_path):
    store = CollectiveIntelligenceStore(db_path=tmp_path / "ci.db")

    store.record_result(
        task_goal="Refactor auth middleware for retries",
        context_excerpt="python backend",
        summary="Completed with better retry policy",
        status="completed",
        duration_seconds=12.0,
        toolsets=["terminal", "file"],
    )
    store.record_result(
        task_goal="Investigate flaky auth test",
        context_excerpt="pytest",
        summary="Found race condition in fixture",
        status="completed",
        duration_seconds=7.0,
        toolsets=["terminal"],
    )

    matches = store.find_relevant("auth retries", limit=5)
    assert len(matches) >= 1
    assert any("auth" in m["task_goal"].lower() for m in matches)


def test_collective_store_returns_empty_on_blank_query(tmp_path):
    store = CollectiveIntelligenceStore(db_path=tmp_path / "ci.db")
    assert store.find_relevant("", limit=5) == []
