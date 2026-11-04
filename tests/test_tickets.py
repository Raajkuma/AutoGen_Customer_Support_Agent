import pytest

import tickets


def _log(**kw):
    base = dict(customer_name="Priya", order_id="ORD-1", complaint="late", category="Shipping", urgency="Low",
                sentiment="Calm", kb_sources="shipping_policy.md", response="ok", confidence="High")
    return tickets.log_ticket(**{**base, **kw})


def test_ids_increment_and_csv_export_matches_columns():
    assert _log()["ticket_id"] == "TKT-0001" and _log()["ticket_id"] == "TKT-0002"
    import config, pandas as pd
    exported = pd.read_csv(config.LOG_FILE, dtype=str)
    assert list(exported.columns) == tickets.COLUMNS and len(exported) == 2


@pytest.mark.parametrize("kw,reason", [
    ({"urgency": "High"}, "High urgency"), ({"sentiment": "Angry"}, "Angry customer"),
    ({"kb_sources": "NONE"}, "No matching"), ({"confidence": "Low"}, "Low reply confidence"),
    ({"complaint": "I will call my lawyer"}, "Legal"), ({"complaint": "I am contacting my bank about this"}, "Legal"),
    ({"complaint": "this is a scam"}, "Legal")])
def test_each_escalation_rule_fires(kw, reason):
    out = _log(**kw)
    assert out["status"] == "Escalated" and out["approval"] == "Pending"
    assert any(reason in r for r in out["reasons"])


@pytest.mark.parametrize("complaint", ["I bank with HDFC, can I pay by net banking?", "Where is my order?",
                                       "The courtesy call was nice"])
def test_no_false_positive_escalation(complaint):
    assert _log(complaint=complaint)["status"] == "Resolved"


def test_resolved_ticket_releases_reply_immediately_escalated_does_not():
    ok, esc = _log(response="hello"), _log(urgency="High", response="draft")
    assert tickets.get_ticket(ok["ticket_id"])["final_response"] == "hello"
    assert tickets.get_ticket(esc["ticket_id"])["final_response"] == ""


def test_human_review_flow():
    t1 = _log(urgency="High", response="draft")["ticket_id"]
    assert tickets.review_ticket(t1, "Sam", "draft")["decision"] == "Approved"
    assert tickets.review_ticket(t1, "Sam", "again") is None            # cannot be reviewed twice
    t2 = _log(urgency="High", response="draft")["ticket_id"]
    assert tickets.review_ticket(t2, "Sam", "better text")["decision"] == "Edited"
    t3 = _log(urgency="High")["ticket_id"]
    done = tickets.review_ticket(t3, "Sam", "", reject=True)
    assert done["decision"] == "Rejected" and done["final_response"] == tickets.REJECT_MESSAGE
    assert tickets.get_ticket(t3)["reviewed_by"] == "Sam"


def test_hostile_text_is_stored_verbatim_not_executed():
    evil = "x'); DROP TABLE tickets;--"
    tid = _log(complaint=evil)["ticket_id"]
    assert tickets.get_ticket(tid)["complaint"] == evil and len(tickets.read_all()) == 1
