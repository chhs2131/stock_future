import FinanceDataReader as fdr

ticker='000660'
stt = '2020-01-01'
end = '2020-01-10'

ohlcv = fdr.DataReader(ticker, stt, end)
ohlcv = ohlcv.iloc[:, 0: -1]
print('ohlcv: ', ohlcv.shape)
print('ohlcv: ', ohlcv)
print(ohlcv.dtypes)

ohlcv = ohlcv.values
