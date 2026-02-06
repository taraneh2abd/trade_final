import os
from google import genai

def main():
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model = os.getenv("GEMINI_MODEL")
    print("has_key:", bool(key))
    print("model:", model)

    if not key:
        raise RuntimeError("No API key in env")
    if not model:
        raise RuntimeError("No GEMINI_MODEL in env")

    client = genai.Client(api_key=key)

    resp = client.models.generate_content(
        model=model,
        contents="Reply with exactly: OK"
    )
    print((resp.text or "").strip())

if __name__ == "__main__":
    main()
