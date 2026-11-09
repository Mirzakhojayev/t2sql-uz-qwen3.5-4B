from t2sql.rft import split_answer, sql_key


def test_split_answer_needs_closed_thinking_and_a_final_query():
    done = "<think>\nJoin a and b.\n</think>\n\nUse a join.\n\n```sql\nSELECT 1\n```"
    assert split_answer(done) == ("Join a and b.", "Use a join.\n\n```sql\nSELECT 1\n```")
    assert split_answer("<think>\nstill going") is None
    assert split_answer("<think>\nplan\n</think>\n\nNo query here.") is None
    assert split_answer("<think>\n</think>\n\n```sql\nSELECT 1\n```") is None  # nothing to learn from


def test_sql_key_ignores_case_spacing_and_semicolon():
    assert sql_key("SELECT  a\nFROM t;") == sql_key("select a from t")
