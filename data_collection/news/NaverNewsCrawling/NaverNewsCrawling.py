import requests
from bs4 import BeautifulSoup


def get_request_query(url, params):
    import urllib.parse as urlparse
    params = urlparse.urlencode(params)
    request_query = url + '?' + params + '&'
    return request_query


url = "https://search.naver.com/search.naver"
params = {
    'where':'news',
    'sm':'tab_pge',
    'query':'삼성전자',
    'sort':'0',
    'photo':'0',
    'field':'0',
    'pd':'3',
    'ds':'2022.05.16',
    'de':'2022.05.16',
    'cluster_rank':'31',
    'mynews':'0',
    'office_type':'0',
    'office_section_code':'0',
    'news_office_checked':'',
    'nso':'so:r,p:from20220516to20220516,a:all',
    'start':'4000'
}

# 네이버 검색 요청 진행
request_query = get_request_query(url, params)
print('url:', request_query)
get_data = requests.get(request_query, headers={'User-Agent': 'Mozilla/5.0'})

# 데이터 분류
html = BeautifulSoup(get_data.text, "html.parser").find("body")
news_list = html.find("section").find_all("li")
page_btn = html.select_one(".sc_page_inner").find_all("a")

# 출력
for nnn in news_list:
    print("뉴스기사:", nnn.text)
for bbb in page_btn:
    print("페이지버튼:", bbb.text)
