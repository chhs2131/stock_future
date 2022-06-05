import pandas as pd
from sklearn import preprocessing
import numpy as np
import FinanceDataReader as fdr
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation
from keras import optimizers
import numpy as np
# from tensorflow import set_random_seed
# from util import csv_to_dataset, history_points
import matplotlib.pyplot as plt

print("TensorFlow Version:", tf.__version__)
np.random.seed(4)
tf.random.set_seed(44)


def call_dataset(stock_name, stt='2020-01-01', end='2020-02-01', history_points=50):
    import data_collection.news.NaverNewsCrawling as NaverNewsCrawling

    # 이름으로 종목 정보 가져오기
    df_krx = fdr.StockListing('KRX')
    list_krx = {}
    for i in range(len(df_krx.Name)):
        list_krx[df_krx['Name'][i]] = {"code": df_krx['Symbol'][i], "sector": df_krx['Sector'][i]}

    ticker = list_krx[stock_name]["code"]
    print(stock_name, "의 종목코드는 ", ticker)

    # OHLCV 값 가져오기
    ohlcv = fdr.DataReader(ticker, stt, end)
    ohlcv = ohlcv.iloc[:, 0:-1]
    # ohlcv = ohlcv.loc[:,['Open','Close','Volume']]  # 시가, 종가, 거래량만 가져오기
    # print('ohlcv: ', ohlcv)

    # 나스닥 지수 (종가) 가져오기
    nasdaq = fdr.DataReader('IXIC', stt, end)
    nasdaq = nasdaq.loc[:, ['Close']]
    nasdaq.rename(columns={"Close": "IXIC_Close"}, inplace=True)  # 컬럼명 바꾸기
    # print('nasdaq: ', nasdaq)

    # 업종별 등락률 가져오기
    # 흠...

    # 뉴스 갯수 가져오기 (넘느려서 일단 주석처리)
    nnc = NaverNewsCrawling()
    news_amount = nnc.get_news_amount_everyday_df(stock_name, stt, end)
    # print(news_amount)

    # 일자별로 데이터 합치기
    data_result = ohlcv.join(nasdaq).join(news_amount)
    data_result = data_result.fillna(method='ffill')  # 값(NaN)이 없는경우 이전일에서 데이터를 가져오도록함.
    print(data_result)
    data_result = data_result.values

    # 데이터를 0~1 범위로 스케일링
    data_normalizer = preprocessing.MinMaxScaler()
    data_normalized = data_normalizer.fit_transform(data_result)

    # 살펴볼 일수(window size = history_points) 단위로 데이터를 저장한다.
    ohlcv_histories_normalized = np.array(
        [data_normalized[i:i + history_points].copy() for i in range(len(data_normalized) - history_points)])

    # y_data (시초가)
    next_day_open_values_normalized = np.array(
        [data_normalized[:, 0][i + history_points].copy() for i in range(len(data_normalized) - history_points)])
    next_day_open_values_normalized = np.expand_dims(next_day_open_values_normalized, -1)  # 1XN 벡터 -> NX1 벡터로

    next_day_open_values = np.array(
        [data_result[:, 0][i + history_points].copy() for i in range(len(data_result) - history_points)])
    next_day_open_values = np.expand_dims(next_day_open_values, -1)  # 1XN 벡터 -> NX1 벡터로

    # 종가 데이터 기록
    close_values = np.array(
        [data_result[:, 1][i + history_points].copy() for i in range(len(data_result) - history_points)])
    close_values = np.expand_dims(close_values, -1)  # 1XN 벡터 -> NX1 벡터로

    # 데이터를 0~1 범위로 스케일링
    y_normalizer = preprocessing.MinMaxScaler()
    y_normalizer.fit(next_day_open_values)

    # 값 반환
    return ohlcv_histories_normalized, next_day_open_values_normalized, next_day_open_values, y_normalizer, close_values


# 기본 정보 입력
history_points = 50
stock_name="LX세미콘"
stt = '2020-01-01'
end = '2022-05-30'

# 데이터 가져오기
ohlcv_histories, next_day_open_values, unscaled_y, y_normaliser, close_values = call_dataset(
    stock_name=stock_name, stt=stt, end=end, history_points=history_points)

train_ratio = 0.7
n = int(ohlcv_histories.shape[0] * train_ratio)

ohlcv_train = ohlcv_histories[-n:-1]
y_train = next_day_open_values[-n:-1]

ohlcv_test = ohlcv_histories[:ohlcv_histories.shape[0]-n]
y_test = next_day_open_values[:ohlcv_histories.shape[0]-n]

unscaled_y_test = unscaled_y[:ohlcv_histories.shape[0]-n]

print('ohlcv_train.shape: ', ohlcv_train.shape)
print('ohlcv_test.shape: ',ohlcv_test.shape)

# 학습 진행
lstm_input = Input(shape=(history_points, 7), name='lstm_input')
x = LSTM(50, name='lstm_0')(lstm_input)
x = Dropout(0.2, name='lstm_dropout_0')(x)
x = Dense(64, name='dense_0')(x)  # 64
x = Activation('sigmoid', name='sigmoid_0')(x)
x = Dense(1, name='dense_1')(x)
output = Activation('linear', name='linear_output')(x)

model = Model(inputs=lstm_input, outputs=output)
adam = tf.keras.optimizers.Adam(lr=0.0005)
model.compile(optimizer=adam, loss='mse')
model.fit(x=ohlcv_train, y=y_train, batch_size=32, epochs=2000, shuffle=True, validation_split=0.1)  # 반복학습 회수 = epochs.

# evaluation
y_test_predicted = model.predict(ohlcv_test)
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
y_predicted = model.predict(ohlcv_histories)
y_predicted = y_normaliser.inverse_transform(y_predicted)

assert unscaled_y_test.shape == y_test_predicted.shape
real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
print(scaled_mse)

# 학습 모델 저장
model.save(f'basic_model.h5')


# 그래프 그리기 (실제가격과 예측가격)
plt.gcf().set_size_inches(22, 15, forward=True)

start = 0
end = -1
real = plt.plot(unscaled_y[start:end], label='real')
pred = plt.plot(y_predicted[start:end], label='predicted')

plt.legend(['Real', 'Predicted'])
plt.title(stock_name + ' Using LSTM by TGG')
plt.show()

# 적합도 확인
col_name = ['real', 'pred', 'close']
real, pred, clo = pd.DataFrame(unscaled_y[start:end]), pd.DataFrame(y_predicted[start:end]), pd.DataFrame(close_values[start:end])
foo = pd.concat([real, pred, clo], axis = 1)
foo.columns = col_name
foo.corr()
