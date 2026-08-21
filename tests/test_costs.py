from __future__ import annotations

import pytest
from PIL import Image

from housefinder.costs import BudgetExceeded, CostLedger, calculate_cost, estimate_image_tokens


def test_patch_token_estimate() -> None:
    assert estimate_image_tokens(1024, 1024) == 1024
    assert estimate_image_tokens(1025, 33) == 66


def test_luna_cost_is_below_one_cent_for_compact_call() -> None:
    cost = calculate_cost("gpt-5.6-luna", input_tokens=8_000, output_tokens=1_000)
    assert cost < 0.01


def test_budget_blocks_expensive_call_before_execution() -> None:
    ledger = CostLedger(budget_usd=0.01)
    images = [Image.new("RGB", (2048, 2048)) for _ in range(4)]
    with pytest.raises(BudgetExceeded):
        ledger.ensure_affordable(
            "Terra",
            "gpt-5.6-terra",
            images,
            "vergelijk kandidaten",
            max_output_tokens=1_200,
        )


def test_ledger_roundtrip() -> None:
    ledger = CostLedger(budget_usd=0.04)
    ledger.record("test", "gpt-5.6-luna", 1000, 200)
    restored = CostLedger.from_dict(ledger.to_dict(), budget_usd=0.04)
    assert restored.total_usd == pytest.approx(ledger.total_usd)
