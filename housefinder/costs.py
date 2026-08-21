from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

from PIL import Image

from .config import MODEL_PRICING


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class CostEntry:
    label: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    estimated: bool = False


@dataclass
class CostLedger:
    budget_usd: float
    entries: list[CostEntry] = field(default_factory=list)

    @property
    def total_usd(self) -> float:
        return sum(entry.cost_usd for entry in self.entries)

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.budget_usd - self.total_usd)

    def estimate_call(
        self,
        model: str,
        images: Iterable[Image.Image],
        text: str,
        max_output_tokens: int,
    ) -> tuple[int, int, float]:
        image_tokens = sum(estimate_image_tokens(image.width, image.height) for image in images)
        text_tokens = max(1, math.ceil(len(text) / 4))
        input_tokens = image_tokens + text_tokens
        output_tokens = max_output_tokens
        return input_tokens, output_tokens, calculate_cost(model, input_tokens, output_tokens)

    def ensure_affordable(
        self,
        label: str,
        model: str,
        images: Iterable[Image.Image],
        text: str,
        max_output_tokens: int,
    ) -> float:
        _, _, projected = self.estimate_call(model, images, text, max_output_tokens)
        if self.total_usd + projected > self.budget_usd + 1e-9:
            raise BudgetExceeded(
                f"{label} zou naar schatting ${projected:.4f} kosten; "
                f"er is nog ${self.remaining_usd:.4f} beschikbaar."
            )
        return projected

    def record(
        self,
        label: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        estimated: bool = False,
    ) -> CostEntry:
        entry = CostEntry(
            label=label,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=calculate_cost(model, input_tokens, output_tokens),
            estimated=estimated,
        )
        self.entries.append(entry)
        return entry

    def to_dict(self) -> dict:
        return {
            "budget_usd": self.budget_usd,
            "entries": [asdict(entry) for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, value: dict | None, budget_usd: float) -> CostLedger:
        if not value:
            return cls(budget_usd=budget_usd)
        return cls(
            budget_usd=budget_usd,
            entries=[CostEntry(**item) for item in value.get("entries", [])],
        )


def estimate_image_tokens(width: int, height: int) -> int:
    """GPT-5.6 original/auto detail uses 32 by 32 pixel patches."""
    return math.ceil(width / 32) * math.ceil(height / 32)


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in MODEL_PRICING:
        raise KeyError(f"Geen prijsconfiguratie voor model {model!r}.")
    pricing = MODEL_PRICING[model]
    return (
        input_tokens * pricing.input_per_million + output_tokens * pricing.output_per_million
    ) / 1_000_000
