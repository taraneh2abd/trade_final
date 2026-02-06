import os
from google import genai

def main():
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) first")

    client = genai.Client(api_key=key)

    models = list(client.models.list())
    print("models_count:", len(models))

    for m in models[:50]:  # فعلاً 50 تای اول
        name = getattr(m, "name", None)
        methods = getattr(m, "supported_generation_methods", None)
        print("name:", name, "| methods:", methods)

if __name__ == "__main__":
    main()
