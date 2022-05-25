from pykiwoom.kiwoom import *

kiwoom = Kiwoom()
kiwoom.CommConnect(block=True)
account_num = kiwoom.GetLoginInfo("ACCOUNT_CNT")        # 전체 계좌수
accounts = kiwoom.GetLoginInfo("ACCNO")                 # 전체 계좌 리스트
user_id = kiwoom.GetLoginInfo("USER_ID")                # 사용자 ID
user_name = kiwoom.GetLoginInfo("USER_NAME")            # 사용자명
keyboard = kiwoom.GetLoginInfo("KEY_BSECGB")            # 키보드보안 해지여부
firewall = kiwoom.GetLoginInfo("FIREW_SECGB")           # 방화벽 설정 여부
kospi = kiwoom.GetCodeListByMarket('0')
kosdaq = kiwoom.GetCodeListByMarket('10')
etf = kiwoom.GetCodeListByMarket('8')
name = kiwoom.GetMasterCodeName("005930")
state = kiwoom.GetConnectState()
if state == 0:
    print("미연결")
elif state == 1:
    print("연결완료")
stock_cnt = kiwoom.GetMasterListedStockCnt("005930")
감리구분 = kiwoom.GetMasterConstruction("005930")
상장일 = kiwoom.GetMasterListedStockDate("005930")
전일가 = kiwoom.GetMasterLastPrice("005930")

print("블록킹 로그인 완료")
print(account_num)
print(accounts)
print(user_id)
print(user_name)
print(keyboard)
print(firewall)
print(len(kospi), kospi)
print(len(kosdaq), kosdaq)
print(len(etf), etf)
print(name)
print("삼성전자 상장주식수: ", stock_cnt)
print(감리구분)
print(상장일)
print(type(상장일))        # datetime.datetime 객체
print(int(전일가))
print(type(전일가))
