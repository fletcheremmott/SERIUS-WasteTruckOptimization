import requests
import json

url = "https://api-open.data.gov.sg/v2/real-time/api/rainfall"

print("Hello World Test for data.gov.sg weather data")
print("URL : " + url)
try:
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    formatted_json = json.dumps(data, indent=4)
    print(formatted_json)

except requests.exceptions.RequestException as e:
    print(f"An error occured {e}")