# tests/test_triage.py

import pytest
from fastapi.testclient import TestClient
from app.main import app, graph

client = TestClient(app)


def test_health_check():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.parametrize(
    "ticket_text, expected_issue, expected_order",
    [
        ("I'd like a refund for order ORD1001.", "refund_request", "ORD1001"),
        ("My Bluetooth speaker (ORD1002) has not arrived yet.", "late_delivery", "ORD1002"),
        ("The smart watch I got (ORD1004) is not working.", "defective_product", "ORD1004"),
        ("Wrong item shipped for order ORD1006.", "wrong_item", "ORD1006"),
        ("one sleeve is missing for ORD1005", "missing_item", "ORD1005"),
    ],
)
def test_triage_invoke(ticket_text, expected_issue, expected_order):
    r = client.post("/triage/invoke", json={"ticket_text": ticket_text})
    assert r.status_code == 200, r.text

    data = r.json()
    assert data["issue_type"] == expected_issue
    assert data["order_id"] == expected_order
    assert "reply_text" in data
    assert len(data["messages"]) > 0  # graph messages exist


def test_missing_order_id():
    # no ORD#### present
    r = client.post("/triage/invoke", json={"ticket_text": "please help with my purchase"})
    assert r.status_code == 400
    assert "order_id missing" in r.json()["detail"].lower()
