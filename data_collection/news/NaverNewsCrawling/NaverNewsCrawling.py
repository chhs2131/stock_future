import requests
import datetime
from bs4 import BeautifulSoup
import pandas as pd


def get_request_query(url, params):
    import urllib.parse as urlparse
    params = urlparse.urlencode(params)
    request_query = url + '?' + params + '&'
    return request_query


class NaverNewsCrawling:
    __url = "https://search.naver.com/search.naver"

    def get_news_amount(self, keyword, date_start, date_end):
        # return : int (검색된 뉴스 페이지 수)
        params = {'where': 'news', 'sm': 'tab_pge', 'sort': '0', 'photo': '0', 'field': '0', 'pd': '3',
                  'cluster_rank': '31', 'mynews': '0', 'office_type': '0', 'office_section_code': '0',
                  'news_office_checked': '', 'nso': 'so:r,p:from20220516to20220516,a:all', 'start': '4000',
                  'query': keyword, 'ds': date_start, 'de': date_end}

        request_query = get_request_query(self.__url, params)
        # print('url:', request_query)
        get_data = requests.get(request_query, headers={'User-Agent': 'Mozilla/5.0'})

        # 데이터 분류
        try:
            html = BeautifulSoup(get_data.text, "html.parser").find("body")
            news_list = html.find("section").find_all("li")
            page_btn = html.select_one(".sc_page_inner").find_all("a")

            # 출력
            news_amount = int(page_btn[len(page_btn) - 1].text)
        except Exception as e:
            news_amount = 0
        # if date_start == date_end:
        #     print(date_start, "마지막 페이지:", news_amount)
        # else:
        #     print(date_start, "~", date_end, "마지막 페이지:", news_amount)
        return news_amount

    def get_news_amount_everyday(self, keyword, date_start, date_end):
        # 시작일(date_start) 부터 종료일(date_end)까지 매일 keyword에 해당하는 기사가 몇 개 나왔는지 확인합니다.
        date_start = datetime.datetime.strptime(date_start, "%Y.%m.%d")
        date_end = datetime.datetime.strptime(date_end, "%Y.%m.%d")

        # 시작일부터 종료일까지 매일 get_news_amount 실행하게 동작
        amount_list = {}
        searching_date = date_start
        while searching_date <= date_end:
            d = searching_date.strftime("%Y.%m.%d")
            na = self.get_news_amount(keyword, d, d)  # 특정일자에 뉴스가 총 몇페이지 검색됬는지 반환합니다.(int)
            # print(searching_date.strftime("%Y년 %m월 %d일  페이지:"), na)
            amount_list[searching_date.strftime("%Y-%m-%d")] = na  # 날짜별로 페이지 수를 저장합니다.

            # 다음날로 넘김
            searching_date += datetime.timedelta(1)
        return amount_list

    def get_news_amount_everyday_df(self, keyword, date_start, date_end):
        # 시작일(date_start) 부터 종료일(date_end)까지 매일 keyword에 해당하는 기사가 몇 개 나왔는지 확인합니다. (반환 데이터프레임)
        date_start = datetime.datetime.strptime(date_start, "%Y.%m.%d")
        date_end = datetime.datetime.strptime(date_end, "%Y.%m.%d")

        # 시작일부터 종료일까지 매일 get_news_amount 실행하게 동작
        date_list = []
        pagenum_list = []
        searching_date = date_start
        while searching_date <= date_end:
            d = searching_date.strftime("%Y.%m.%d")
            na = self.get_news_amount(keyword, d, d)  # 특정일자에 뉴스가 총 몇페이지 검색됬는지 반환합니다.(int)
            # print(searching_date.strftime("%Y년 %m월 %d일  페이지:"), na)
            date_list.append(searching_date)
            pagenum_list.append(na)

            # 다음날로 넘김
            searching_date += datetime.timedelta(1)

        amount_list = {"Date": date_list, "pagenum": pagenum_list}
        amount_df = pd.DataFrame(amount_list, columns=["Date", "pagenum"])
        amount_df = amount_df.set_index('Date')
        return amount_df


# main
nnc = NaverNewsCrawling()
search_keyword = 'LX세미콘'  # 검색할 단어
search_date_start = '2022.01.01'  # 시작일
search_date_end = '2022.01.31'  # 종료일
# al = nnc.get_news_amount_everyday(search_keyword, search_date_start, search_date_end)
#
# # 출력
# for a_key, a_value in al.items():
#     print(a_key, "  페이지 수:", a_value)

al = nnc.get_news_amount_everyday_df(search_keyword, search_date_start, search_date_end)
print(al)
