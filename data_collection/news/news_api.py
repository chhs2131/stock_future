import json
import requests

# readline_test.py
f = open("./.apikey", 'r')
apikey = f.readline()
print(apikey)
f.close()

url = 'https://newsapi.org/v2/everything?q=1&language=kr&from=2022-05-09&to=2022-05-10&sortBy=publishedAt&apiKey=' + apikey
get_data = requests.get(url)
json_data = json.loads(get_data.content)
print(json.dumps(json_data, indent=4, ensure_ascii=False))
