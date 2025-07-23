import requests

url = "http://localhost:8000/predict"
data = {
    "headlines": [
        "Nifty climbs above 25,000 for the first time",
        "Investors cautious ahead of Fed meeting"
    ]
}

response = requests.post(url, json=data)
json_response = response.json()

# Correctly access the list inside the "results" key
results = json_response["results"]

print("📊 Prediction Results:")
for line in results:
    print(f"📰 {line}")
