from pathlib import Path

from agent.skills_evolution_store import (
    SkillsEvolutionStore,
    compute_unified_diff,
)


def test_record_and_query_lineage(tmp_path):
    db = tmp_path / "evolution.db"
    store = SkillsEvolutionStore(db_path=db)

    v1 = store.record_version(
        skill_name="demo-skill",
        skill_path="/tmp/demo-skill",
        action="create",
        snapshot={"SKILL.md": "hello"},
        diff_text="+hello",
    )
    v2 = store.record_version(
        skill_name="demo-skill",
        skill_path="/tmp/demo-skill",
        action="edit",
        parent_version_id=v1,
        snapshot={"SKILL.md": "hello world"},
        diff_text="+world",
    )

    assert store.latest_version("demo-skill") == v2

    skills = store.list_skills(limit=10)
    assert skills[0]["skill_name"] == "demo-skill"
    assert skills[0]["version_count"] == 2

    lineage = store.get_skill_lineage("demo-skill")
    assert len(lineage) == 2
    assert lineage[1]["parent_version_id"] == v1

    rec = store.get_version(v2)
    assert rec is not None
    assert rec.skill_name == "demo-skill"
    assert rec.snapshot["SKILL.md"] == "hello world"


def test_compute_unified_diff_handles_added_and_removed_files():
    before = {
        "SKILL.md": "line1\nline2\n",
        "references/a.md": "old\n",
    }
    after = {
        "SKILL.md": "line1\nline2 changed\n",
        "references/b.md": "new\n",
    }

    diff = compute_unified_diff(before, after)
    assert "a/SKILL.md" in diff
    assert "b/SKILL.md" in diff
    assert "a/references/a.md" in diff
    assert "b/references/b.md" in diff
