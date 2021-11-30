import requests
import json

api_key = "pub_228565b25933ad33e2f1aae23c17b23b6705"
response = requests.get("https://newsdata.io/api/1/news?apikey=pub_228565b25933ad33e2f1aae23c17b23b6705&category=business&language=en")
print(response.status_code)
response = response.json()
content = dict()
results = response['results']
i = 0
for result in results:
    if(result['link'] == None) or (result['content'] == None):
        continue
    else:
        content[i] = result['content']
        i = i+1

for k in content.keys():
    print(content[k][:200])
    print(" ")