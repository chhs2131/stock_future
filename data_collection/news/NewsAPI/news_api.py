import json
import requests

# readline_test.py
f = open(".apikey", 'r')
apikey = f.readline()
print(apikey)
f.close()

url = 'https://newsapi.org/v2/everything?' \
      'q=코로나' \
      '&from=2022-05-10' \
      '&to=2022-05-16' \
      '&pageSize=0' \
      '&apiKey=' + apikey
get_data = requests.get(url)
json_data = json.loads(get_data.content)
print(json.dumps(json_data, indent=4, ensure_ascii=False))
