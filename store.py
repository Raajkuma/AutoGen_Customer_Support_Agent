"""E-commerce data layer (SQLite): products, customers' orders."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import db

CUSTOMERS = ["Priya", "Arjun"]

PRODUCTS = [
    ("P1", "Running Shoes", 2499, "👟", "Lightweight daily trainers"),
    ("P2", "Wireless Earbuds", 1999, "🎧", "20-hour battery, IPX4"),
    ("P3", "Countertop Blender", 3499, "🥤", "600W, 1.5 L jar"),
    ("P4", "Steel Water Bottle", 599, "🍶", "750 ml, keeps cold for 24 h"),
    ("P5", "Commuter Backpack", 1799, "🎒", "25 L, laptop sleeve"),
    ("P6", "LED Desk Lamp", 1299, "💡", "Dimmable, USB powered"),
    ("P7", "Yoga Mat", 899, "🧘", "6 mm non-slip"),
    ("P8", "Smart Watch", 4999, "⌚", "Heart-rate and sleep tracking"),
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS products(id TEXT PRIMARY KEY, name TEXT, price INTEGER, emoji TEXT, blurb TEXT);
CREATE TABLE IF NOT EXISTS orders(
  order_id TEXT PRIMARY KEY, customer TEXT, created_at TEXT, expected_delivery TEXT,
  status TEXT, total INTEGER, items TEXT);
"""


def _insert_order(c, order_id: str, customer: str, days_ago: int, eta_days: int, status: str, items: list[dict]) -> None:
    created = datetime.now() - timedelta(days=days_ago)
    total = sum(i["price"] * i["qty"] for i in items)
    c.execute(
        "INSERT INTO orders VALUES (?,?,?,?,?,?,?)",
        (order_id, customer, created.strftime("%Y-%m-%d"),
         (created + timedelta(days=eta_days)).strftime("%Y-%m-%d"),
         status, total, json.dumps(items)),
    )


def init_db() -> None:
    """Create tables and seed demo data once."""
    with db.connect() as c:
        c.executescript(_SCHEMA)
        if c.execute("SELECT 1 FROM products").fetchone():
            return
        c.executemany("INSERT INTO products VALUES (?,?,?,?,?)", PRODUCTS)
        item = lambda pid, qty=1: {  # noqa: E731
            "id": pid, "qty": qty,
            "name": next(p[1] for p in PRODUCTS if p[0] == pid),
            "price": next(p[2] for p in PRODUCTS if p[0] == pid)}
        _insert_order(c, "ORD-4498", "Priya", 9, 5, "Delivered", [item("P1")])
        _insert_order(c, "ORD-4501", "Priya", 4, 5, "Delivered", [item("P3")])
        _insert_order(c, "ORD-4512", "Priya", 10, 5, "Shipped", [item("P2"), item("P4")])
        _insert_order(c, "ORD-4520", "Arjun", 1, 5, "Processing", [item("P8")])


def list_products() -> list[dict]:
    with db.connect() as c:
        return [dict(r) for r in c.execute("SELECT * FROM products ORDER BY id")]


def create_order(customer: str, cart: dict[str, int]) -> str:
    """cart maps product id -> quantity. Returns the new order id."""
    prods = {p["id"]: p for p in list_products()}
    items = [{"id": k, "qty": q, "name": prods[k]["name"], "price": prods[k]["price"]}
             for k, q in cart.items() if q > 0 and k in prods]
    if not items:
        raise ValueError("Cart is empty")
    with db.connect() as c:
        last = c.execute("SELECT MAX(CAST(SUBSTR(order_id,5) AS INTEGER)) FROM orders").fetchone()[0]
        order_id = f"ORD-{(last or 4500) + 1}"
        _insert_order(c, order_id, customer, 0, 5, "Processing", items)
    return order_id


def list_orders(customer: str) -> list[dict]:
    with db.connect() as c:
        rows = c.execute("SELECT * FROM orders WHERE customer=? ORDER BY created_at DESC, order_id DESC",
                         (customer,)).fetchall()
    return [{**dict(r), "items": json.loads(r["items"])} for r in rows]


def get_order(order_id: str, customer: str) -> dict | None:
    """Only returns the order if it belongs to this customer."""
    return next((o for o in list_orders(customer) if o["order_id"] == order_id), None)


def order_context(order: dict | None) -> str:
    """Verified, system-generated order facts handed to the agents."""
    if not order:
        return "none"
    items = ", ".join(f"{i['qty']}x {i['name']}" for i in order["items"])
    return (f"{order['order_id']} | items: {items} | total: Rs {order['total']} | "
            f"placed: {order['created_at']} | expected delivery: {order['expected_delivery']} | "
            f"status: {order['status']} | today: {datetime.now():%Y-%m-%d}")
