import requests
import json

api_key = "pub_228565b25933ad33e2f1aae23c17b23b6705"
response = requests.get("https://newsdata.io/api/1/news?apikey=pub_228565b25933ad33e2f1aae23c17b23b6705&category=business&language=en")
print(response.status_code)
response = response.json()

results = response['results']
for result in results:
    if(result['link'] == None):
        continue
    else:

    
        print(result['title'])

        print(result['description'])

        print(result['link'])

        print("")
        print("")
        print("")
        print("")