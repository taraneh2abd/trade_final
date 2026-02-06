import requests

# api-free آشغال
# sk-p4EnCbF77KDbjZc5bbJCcC8dfe7w2

API_KEYS = [
    "sk-RrrmNQIbMVcfmg79VcqIcuzy84RqIRcNtONmHlBUZ1TcqH99",
    "sk-hnHJsWH2pyPGovdL7M5hNgZ1UlaRxSFIEGNj17p5iAAyQ0y8",
    "sk-hVrWCsICZ5yUyQ1HsJSZL9PI0Sr3X1vG3bfxuIvnLB8iaMAi", ## we are using this
    "sk-jtN3PI3R0vywtGmoSakTX9TPNgjIAbkkFHvYZSbX3e5wbtoZ"
]

URL = "https://api.chatanywhere.tech/v1/chat/completions"

for i, key in enumerate(API_KEYS, 1):
    try:
        response = requests.post(
            URL,
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "سلام"}],
                "max_tokens": 10
            },
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"\033[92m✅ Key {i}: active\033[0m")
        elif response.status_code == 429:
            print(f"\033[93m⚠️  Key {i}:limitation\033[0m")
        else:
            print(f"\033[91m❌ Key {i}: error ({response.status_code})\033[0m")
            
    except Exception as e:
        print(f"\033[91m❌ Key {i}: connection error\033[0m")