"""Brain front-end: natural-language command -> wall plan spec.

Implements Layer 1 (Strategic Planning):
- Translates natural-language user commands into structured PlanSpec JSON.
- Uses Gemini LLM when available (via google-genai SDK).
- Gracefully falls back to deterministic regex parser (parse_command) when offline.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

from .prompts import PLANNER_SYSTEM_PROMPT


@dataclass
class PlanSpec:
    bricks_per_row: int = 3
    rows: int = 3
    bond_pattern: str = "running"


# Legacy prompt template for compatibility
LLM_PROMPT = PLANNER_SYSTEM_PROMPT


def parse_command(text: str) -> PlanSpec:
    """Deterministic stand-in for the LLM. Understands 'WxH', 'W by H',
    '<n> rows' style fragments; falls back to the 3x3 default."""
    m = re.search(r"(\d+)\s*(?:x|by|\*)\s*(\d+)", text)
    if m:
        w, h = int(m.group(1)), int(m.group(2))
        if 1 <= w <= 6 and 1 <= h <= 5:
            return PlanSpec(bricks_per_row=w, rows=h)
    m = re.search(r"(\d+)\s*rows?", text, re.IGNORECASE)
    if m and 1 <= int(m.group(1)) <= 5:
        spec = PlanSpec(rows=int(m.group(1)))
        m2 = re.search(r"(\d+)\s*(?:bricks|wide|per row)", text, re.IGNORECASE)
        if m2 and 1 <= int(m2.group(1)) <= 6:
            spec.bricks_per_row = int(m2.group(1))
        return spec
    return PlanSpec()


def llm_plan(text: str,
             model: str = "gemini-2.5-flash",
             api_key: Optional[str] = None) -> PlanSpec:
    """Translates user natural language command to PlanSpec via Gemini LLM.
    
    If the API key is unavailable, network fails, or google-genai is not installed,
    falls back cleanly to deterministic regex parser.
    """
    resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not resolved_key:
        print("[Brain Planner] Note: GEMINI_API_KEY not set; using deterministic parser.")
        return parse_command(text)

    try:
        from google import genai
        client = genai.Client(api_key=resolved_key)
        
        prompt = (
            f"{PLANNER_SYSTEM_PROMPT}\n\n"
            f"User command: {text}\n"
            f"Return ONLY valid JSON matching the schema."
        )

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "temperature": 0.0,
            }
        )

        raw_text = response.text.strip()
        data = json.loads(raw_text)
        w = int(data.get("bricks_per_row", 3))
        h = int(data.get("rows", 3))
        pattern = str(data.get("bond_pattern", "running"))
        print(f"[Brain Planner] Gemini planned: {w} bricks/row, {h} rows ({pattern} bond).")
        return PlanSpec(bricks_per_row=w, rows=h, bond_pattern=pattern)

    except Exception as e:
        print(f"[Brain Planner Warning] Gemini planning failed: {e}. Falling back to deterministic parser.")
        return parse_command(text)
