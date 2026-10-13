"""The support agent's tools, backed by small in-memory tables."""

from __future__ import annotations

from dataclasses import dataclass

from agent_doctor import trace_tool
from agent_doctor.testing import sleep

ORDERS = {
    "1234": {"status": "shipped", "carrier": "BlueDart", "eta": "Friday"},
    "5678": {"status": "shipped", "carrier": "Delhivery", "eta": "Monday"},
    "9012": {"status": "packed", "carrier": "Ekart", "eta": "Wednesday"},
}
CUSTOMERS = {
    "C-101": {"name": "Asha Rao", "tier": "gold", "region": "IN-KA"},
    "C-202": {"name": "Vikram Shah", "tier": "silver", "region": "IN-MH"},
}
INVENTORY = {"SKU-RED-42": 12, "SKU-BLU-40": 0, "SKU-GRN-38": 3}
REFUND_ARTICLE = {
    "article_id": "KB-12", "title": "Refund policy", "refund_days": 30,
    "snippet": "Unused items can be returned for a full refund.",
}


class ServiceDown(Exception):
    pass


@dataclass
class ToolFaults:
    tracking_failures: int = 0   # get_tracking fails this many times, then works
    tracking_down: bool = False  # get_tracking always fails
    slow_inventory: bool = False


class Tools:
    """One instance per run, so each run gets its own fault state."""

    def __init__(self, faults: ToolFaults | None = None):
        self.faults = faults or ToolFaults()
        self._tracking_calls = 0

    @trace_tool
    def search_orders(self, order_id: str) -> dict:
        sleep(0.12)
        order = ORDERS.get(order_id)
        if order is None:
            return {"order_id": order_id, "found": False}
        items = [{"line": i, "sku": f"SKU-{i:03d}", "name": f"Cotton crew socks, pack of 3, colour {i}",
                  "qty": 1, "price_inr": 499 + 10 * i, "gift_wrap": False, "seller": "ShopMart Retail"}
                 for i in range(1, 41)]
        return {"order_id": order_id, "found": True, **order, "items": items}

    @trace_tool
    def get_tracking(self, order_id: str, carrier: str) -> dict:
        self._tracking_calls += 1
        sleep(0.15)
        if self.faults.tracking_down or self._tracking_calls <= self.faults.tracking_failures:
            sleep(0.35)
            raise ServiceDown(f"{carrier} tracking API timed out after 500 ms")
        order = ORDERS[order_id]
        scans = [{"time": f"2026-09-{20 + i // 3:02d}T{8 + i % 3 * 4:02d}:00",
                  "place": ["Mumbai hub", "Pune sort centre", "Bengaluru hub"][i % 3],
                  "event": "in transit"} for i in range(18)]
        return {"order_id": order_id, "carrier": carrier, "eta": order["eta"], "scans": scans}

    @trace_tool
    def search_kb(self, query: str) -> dict:
        sleep(0.2)
        return {"query": query, "results": [REFUND_ARTICLE]}

    @trace_tool
    def read_article(self, article_id: str) -> dict:
        sleep(0.15)
        return {**REFUND_ARTICLE, "body": "Items can be returned within the refund window. " * 6}

    @trace_tool
    def get_customer(self, customer_id: str) -> dict:
        sleep(0.5)
        return {"customer_id": customer_id, **CUSTOMERS[customer_id]}

    @trace_tool
    def get_promotions(self, locale: str) -> dict:
        sleep(0.6)
        return {"locale": locale, "promo_code": "FESTIVE10", "discount_pct": 10}

    @trace_tool
    def check_inventory(self, sku: str) -> dict:
        sleep(4.5 if self.faults.slow_inventory else 0.3)
        return {"sku": sku, "in_stock": INVENTORY.get(sku, 0), "warehouse": "BLR-2"}

    @trace_tool(side_effects=True)
    def send_reply(self, customer_id: str, text: str) -> dict:
        sleep(0.03)
        return {"delivered": True}
