# from __future__ import annotations

# import json
# from dataclasses import dataclass
# from typing import Any, Dict, Optional

# import requests


# @dataclass
# class ChatAnywhereClient:
#     """
#     Minimal client for ChatAnywhere OpenAI-compatible endpoint.
#     Uses env/arg API key. DO NOT hardcode keys in code.
#     """
#     api_key: str
#     model: str = "gpt-4o-mini-2024-07-18"
#     base_url: str = "https://api.chatanywhere.tech/v1/chat/completions"
#     timeout_sec: int = 60

#     def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json",
#         }
#         r = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout_sec)
#         r.raise_for_status()
#         return r.json()

#     def generate_text(self, system: str, user: str, temperature: float = 0.2) -> Dict[str, Any]:
#         payload = {
#             "model": self.model,
#             "temperature": float(temperature),
#             "messages": [
#                 {"role": "system", "content": system},
#                 {"role": "user", "content": user},
#             ],
#         }
#         raw = self._post(payload)
#         text = raw["choices"][0]["message"]["content"]
#         return {"text": text, "raw": raw}

#     def generate_json(self, system: str, user: str, temperature: float = 0.0) -> Dict[str, Any]:
#         """
#         Expect the assistant to return a JSON object in message content.
#         We parse it strictly; if it fails, raise ValueError.
#         """
#         out = self.generate_text(system=system, user=user, temperature=temperature)
#         text = out["text"].strip()

#         # Common cleanup: sometimes model wraps in ```json ... ```
#         if text.startswith("```"):
#             text = text.strip("`")
#             # remove optional "json" label
#             text = text.replace("json\n", "", 1).strip()

#         try:
#             parsed = json.loads(text)
#         except Exception as e:
#             raise ValueError(f"LLM did not return valid JSON. Text was:\n{text}") from e

#         return {"text": out["text"], "json": parsed, "raw": out["raw"]}
import os
import requests
import json
from typing import Any, Dict, Optional


class ChatAnywhereClient:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini-2024-07-18"):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.chatanywhere.tech/v1/chat/completions"

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        r = requests.post(self.url, headers=headers, json=payload, timeout=60)

        # ✅ خطای 429/401/... رو با متن کامل نشون بده
        if not r.ok:
            print("\n--- ChatAnywhere HTTP ERROR ---")
            print("status:", r.status_code)
            print("text:", r.text[:2000])
            print("--- END ERROR ---\n")
            r.raise_for_status()

        return r.json()

    def generate_text(self, system: str, user: str, temperature: float = 0.2) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "stream": False,
        }
        raw = self._post(payload)
        content = raw["choices"][0]["message"]["content"]
        return {"text": content, "raw": raw}

    def generate_json(self, system: str, user: str, temperature: float = 0.2) -> Dict[str, Any]:
        out = self.generate_text(system=system, user=user, temperature=temperature)
        text = out["text"]
        # تلاش برای parse JSON
        try:
            j = json.loads(text)
        except Exception:
            # اگر مدل قبل/بعدش حرف زد، JSON رو از داخلش بیرون بکش
            start = text.find("{")
            end = text.rfind("}")
            j = json.loads(text[start:end + 1]) if (start != -1 and end != -1 and end > start) else {}
        return {"text": text, "raw": out["raw"], "json": j}
