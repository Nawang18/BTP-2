"""Brain front-end: natural-language command -> buildable brick layout.

The LLM designs the structure (which bricks go in which row, and where).
The geometry checker in structure_plan.py verifies it; if it is not
buildable, the problems are sent back to the LLM so it can correct itself.
Without an API key, a regex parser handles the simple cases.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

from .prompts import PLANNER_SYSTEM_PROMPT
from ..config import Config
from ..structure_plan import (Layout, rectangular_layout, centered_layout,
                              validate_layout)


@dataclass
class PlanSpec:
    layout: Layout
    source: str = "llm"          # llm | fallback | cli

    @property
    def rows(self) -> int:
        return len(self.layout.rows)

    @property
    def bricks_per_row(self) -> int:
        return max(len(r.bricks) for r in self.layout.rows)

    @property
    def bond_pattern(self) -> str:
        return self.layout.description


LLM_PROMPT = PLANNER_SYSTEM_PROMPT


def parse_command(text: str) -> PlanSpec:
    """Offline fallback. Understands '3x2', '3 bricks then 2 then 1 brick',
    and '3 bricks per row, 2 rows'. Anything else gets a 3x3 wall."""
    m = re.search(r"(\d+)\s*(?:x|by|\*)\s*(\d+)", text, re.IGNORECASE)
    if m:
        w, h = int(m.group(1)), int(m.group(2))
        if 1 <= w <= 6 and 1 <= h <= 5:
            return PlanSpec(rectangular_layout(w, h), "fallback")

    counts = [int(n) for n in re.findall(r"(\d+)\s*(?:full\s+)?bricks?\b", text, re.IGNORECASE)]
    if len(counts) >= 2 and all(1 <= n <= 6 for n in counts):
        return PlanSpec(centered_layout(counts), "fallback")

    m = re.search(r"(\d+)\s*rows?", text, re.IGNORECASE)
    if m and 1 <= int(m.group(1)) <= 5:
        w = counts[0] if counts else 3
        return PlanSpec(rectangular_layout(w, int(m.group(1))), "fallback")

    print("[Brain Planner] Could not understand the command offline; using a default 3x3 wall.")
    return PlanSpec(rectangular_layout(3, 3), "fallback")


def llm_plan(text: str,
             model: str = "gemini-2.5-flash",
             api_key: Optional[str] = None,
             cfg: Optional[Config] = None,
             max_attempts: int = 3) -> PlanSpec:
    cfg = cfg or Config()
    resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not resolved_key:
        print("[Brain Planner] GEMINI_API_KEY not set; using the offline parser.")
        return parse_command(text)

    try:
        from google import genai
        client = genai.Client(api_key=resolved_key)
    except Exception as e:
        print(f"[Brain Planner] Gemini unavailable ({e}); using the offline parser.")
        return parse_command(text)

    prompt = f"{PLANNER_SYSTEM_PROMPT}\n\nUser command: {text}\n"
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"response_mime_type": "application/json", "temperature": 0.2},
            )
            raw = response.text.strip()
            layout = Layout.from_dict(json.loads(raw))
            errors = validate_layout(layout, cfg)
        except Exception as e:
            raw, errors = locals().get("raw", ""), [f"could not read the answer as a layout: {e}"]

        if not errors:
            print(f"[Brain Planner] Gemini designed: {layout.description} "
                  f"({layout.brick_count()} bricks, attempt {attempt})")
            return PlanSpec(layout, "llm")

        print(f"[Brain Planner] Attempt {attempt} rejected by geometry check:")
        for err in errors:
            print(f"    - {err}")
        prompt += ("\n\nYour previous answer was:\n" + raw +
                   "\n\nThe geometry checker rejected it:\n- " + "\n- ".join(errors) +
                   "\n\nFix these problems and return the corrected JSON only.")

    print("[Brain Planner] Gemini could not produce a buildable layout; using the offline parser.")
    return parse_command(text)
