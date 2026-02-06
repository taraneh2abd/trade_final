import requests

URL = "https://api.chatanywhere.tech/v1/chat/completions"

API_KEY = "sk-RrrmNQIbMVcfmg79VcqIcuzy84RqIRcNtONmHlBUZ1TcqH99"  # <-- کلیدت رو اینجا بگذار (بعداً پاکش کن)

payload = {
    "model": "gpt-3.5-turbo",
    "messages": [
        {"role": "user", "content": "Reply with exactly: OK"}
    ],
    "temperature": 0,
}

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

r = requests.post(URL, json=payload, headers=headers, timeout=60)
print("status:", r.status_code)
print(r.text)
r.raise_for_status()

data = r.json()
print("\nassistant:", data["choices"][0]["message"]["content"])
