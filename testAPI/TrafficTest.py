import requests
import json

# url = "https://datamall2.mytransport.sg/ltaodataservice/EstTravelTimes"

print("Hello World Test for Land Transport Authority traffic data")
print("URL : " + url)
try:
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    print(data)

except requests.exceptions.RequestException as e:
    print(f"An error occured {e}")
