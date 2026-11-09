from t2sql.retry import NO_ANSWER, answer_part, feedback, retry_messages

DONE = "<think>\nplan\n</think>\n\nJoin a to b.\n\n```sql\nSELECT 1\n```"
CUT = "<think>\nstill thinking"


def test_answer_part_drops_the_thinking():
    assert answer_part(DONE) == "Join a to b.\n\n```sql\nSELECT 1\n```"
    assert answer_part(CUT) == ""


def test_feedback_by_outcome():
    assert feedback(DONE, True, None, 3) is None
    assert "error" in feedback(DONE, False, 'column "x" does not exist', None)
    assert 'column "x" does not exist' in feedback(DONE, False, 'column "x" does not exist', None)
    assert "no rows" in feedback(DONE, True, None, 0)
    assert "no final SQL" in feedback(CUT, None, None, None)


def test_retry_conversation_keeps_the_prompt_and_adds_two_turns():
    prompt = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    msgs = retry_messages(prompt, CUT, "note")
    assert msgs[:2] == prompt
    assert msgs[2] == {"role": "assistant", "content": NO_ANSWER}
    assert msgs[3] == {"role": "user", "content": "note"}
