import requests
import json

API_URL = "http://127.0.0.1:8000/recommend"

def get_int(prompt, min_val=None, max_val=None):
    while True:
        try:
            value = int(input(prompt).strip())
            if min_val is not None and value < min_val:
                print(f"Enter a value >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"Enter a value <= {max_val}")
                continue
            return value
        except ValueError:
            print("Please enter a valid integer.")

def get_text(prompt):
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("This field cannot be empty.")

payload = {
    "age": get_int("Age: ", 0, 120),
    "gender": get_text("Gender: "),
    "symptoms": get_text("Symptoms: "),
    "severity": get_int("Severity (1-10): ", 1, 10),
    "duration": get_text("Duration: ")
}

print("\nSending request...")
print(json.dumps(payload, indent=2))
print("-" * 50)

try:
    response = requests.post(API_URL, json=payload, timeout=30)
    print(f"Status Code: {response.status_code}")
    print("-" * 50)

    try:
        data = response.json()
        print(json.dumps(data, indent=2))
    except Exception:
        print(response.text)

except requests.exceptions.ConnectionError:
    print("Could not connect to FastAPI server. Make sure uvicorn is running on http://127.0.0.1:8000")
except requests.exceptions.Timeout:
    print("Request timed out.")
except Exception as e:
    print(f"Error: {e}")