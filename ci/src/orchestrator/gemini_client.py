from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from google import genai


@dataclass
class GeminiReply:
    raw_text: str
    json_data: Dict[str, Any]


class GeminiClient:
    def __init__(self, model: str = "gemini-1.5-flash"):
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. This orchestrator requires REAL LLM.")
        self.client = genai.Client(api_key=key)
        self.model = model

    def generate_json(self, system: str, user: str) -> GeminiReply:
        """
        Gemini را مجبور می‌کنیم JSON بدهد.
        """
        contents = [
            {"role": "user", "parts": [{"text": f"SYSTEM:\n{system}\n\nUSER:\n{user}"}]}
        ]
        resp = self.client.models.generate_content(
            model=self.model,
            contents=contents,
        )

        text = (resp.text or "").strip()
        data = self._extract_json(text)
        return GeminiReply(raw_text=text, json_data=data)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        اگر Gemini اطراف JSON حرف اضافه زد، JSON وسطش را می‌کشیم بیرون.
        """
        # تلاش 1: مستقیم
        try:
            return json.loads(text)
        except Exception:
            pass

        # تلاش 2: پیدا کردن اولین {...} بزرگ
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            chunk = text[start : end + 1]
            try:
                return json.loads(chunk)
            except Exception:
                pass

        raise ValueError(f"Gemini did not return valid JSON. Raw:\n{text}")
