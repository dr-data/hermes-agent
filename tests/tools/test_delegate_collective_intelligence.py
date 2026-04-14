from tools import delegate_tool


class DummyStore:
    def __init__(self):
        self.recorded = []

    def find_relevant(self, goal, limit=3):
        return [
            {
                "task_goal": "Refactor auth module",
                "status": "completed",
                "score": 0.9,
                "summary": "Use focused failing tests before touching implementation.",
            }
        ]

    def record_result(self, **kwargs):
        self.recorded.append(kwargs)
        return "ci_test"


def test_build_collective_context_injects_prior_knowledge(monkeypatch):
    store = DummyStore()
    monkeypatch.setattr(delegate_tool, "get_collective_intelligence_store", lambda: store)

    merged = delegate_tool._build_collective_context(
        "Improve auth retries",
        "Existing context block",
    )

    assert "COLLECTIVE PRIOR KNOWLEDGE" in merged
    assert "Refactor auth module" in merged
    assert "Existing context block" in merged


def test_record_collective_result_persists_summary(monkeypatch):
    store = DummyStore()
    monkeypatch.setattr(delegate_tool, "get_collective_intelligence_store", lambda: store)

    task = {
        "goal": "Fix flaky auth tests",
        "context": "pytest and fixtures",
        "toolsets": ["terminal", "file"],
    }
    result = {
        "summary": "Resolved fixture race condition",
        "status": "completed",
        "duration_seconds": 11.5,
    }

    delegate_tool._record_collective_result(task, result)

    assert len(store.recorded) == 1
    payload = store.recorded[0]
    assert payload["task_goal"] == "Fix flaky auth tests"
    assert payload["status"] == "completed"
    assert payload["toolsets"] == ["terminal", "file"]
