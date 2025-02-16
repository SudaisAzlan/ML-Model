import requests

url = "http://127.0.0.1:5000/predict"
data = {"features": [2, 8, 128, 4000]} 
response = requests.post(url, json=data) 
print(response.json()) 

