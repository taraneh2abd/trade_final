import os
from google import genai

key = os.getenv("GEMINI_API_KEY")
if not key:
    raise RuntimeError("GEMINI_API_KEY is not set")

client = genai.Client(api_key=key)
resp = client.models.generate_content(
    model="gemini-1.5-flash",
    contents="Reply with exactly: OK"
)
print(resp.text.strip())
