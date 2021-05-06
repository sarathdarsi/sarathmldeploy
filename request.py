import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'temperature':2, 'pressure':9, 'hummidity':6})

print(r.json())
