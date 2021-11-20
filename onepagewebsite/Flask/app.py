import requests
import json
from flask import Flask, render_template

#api part
api_key = "pub_228565b25933ad33e2f1aae23c17b23b6705"
response = requests.get("https://newsdata.io/api/1/news?apikey=pub_228565b25933ad33e2f1aae23c17b23b6705&category=business&language=en")
print(response.status_code)
response = response.json()
results = response['results']

article_head=[]
article_description=[]
article_link=[]


for result in results:
    if(result['link'] == None):
        continue
    else:
        if(result['description'] == None):
            continue
        else:
            article_head.append(result['title'])
            article_description.append(result['description'])
            article_link.append(result['link'])


length=len(article_head)


#flask integration
app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', article_head=article_head, article_description=article_description, article_link=article_link, length=length)

if __name__ == "__main__":
    app.run(debug=True)