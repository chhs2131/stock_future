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


def call_dataset(stock_name, stt='2020-01-01', end='2021-03-30', history_points=50):
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
    print('ohlcv: ', ohlcv.shape)
    print('ohlcv: ', ohlcv)
    ohlcv = ohlcv.values

    # 나스닥 지수 가져오기
    nasdaq = fdr.DataReader('IXIC', stt, end)
    nasdaq = nasdaq.iloc[:, 0:-1]
    print('nasdaq: ', nasdaq.shape)
    print('nasdaq: ', nasdaq)

    # 업종별 등락률 가져오기
    # 흠...

    # 뉴스 갯수 가져오기

    # 일자별로 데이터 합치기

    # 데이터를 0~1 범위로 스케일링
    data_normalizer = preprocessing.MinMaxScaler()
    data_normalized = data_normalizer.fit_transform(ohlcv)
    print('data_normalized: ', data_normalized.shape)

    # 살펴볼 일수(window size = history_points) 단위로 데이터를 저장한다. (시가, 종가, 고가, 저가, 거래량을 사용하여 다음날 시가를 예측)
    ohlcv_histories_normalized = np.array(
        [data_normalized[i:i + history_points].copy() for i in range(len(data_normalized) - history_points)])
    print('ohlcv_histories_normalized: ', ohlcv_histories_normalized.shape)

    next_day_open_values_normalized = np.array(
        [data_normalized[:, 0][i + history_points].copy() for i in range(len(data_normalized) - history_points)])
    next_day_open_values_normalized = np.expand_dims(next_day_open_values_normalized, -1)  # 1XN 벡터 -> NX1 벡터로

    next_day_open_values = np.array(
        [ohlcv[:, 0][i + history_points].copy() for i in range(len(ohlcv) - history_points)])
    next_day_open_values = np.expand_dims(next_day_open_values, -1)  # 1XN 벡터 -> NX1 벡터로

    y_normalizer = preprocessing.MinMaxScaler()
    y_normalizer.fit(next_day_open_values)

    # 값 반환
    return ohlcv_histories_normalized, next_day_open_values_normalized, next_day_open_values, y_normalizer



print("TensorFlow Version:", tensorflow.__version__)
np.random.seed(4)
tensorflow.random.set_seed(44)
history_points = 50
stock_name="SK하이닉스"

# 데이터 가져오기
ohlcv_histories, next_day_open_values, unscaled_y, y_normaliser = call_dataset(stock_name="SK하이닉스", history_points=history_points)

train_ratio = 0.7
n = int(ohlcv_histories.shape[0] * train_ratio)

ohlcv_train = ohlcv_histories[-n:-1]
y_train = next_day_open_values[-n:-1]

ohlcv_test = ohlcv_histories[:ohlcv_histories.shape[0]-n]
y_test = next_day_open_values[:ohlcv_histories.shape[0]-n]

unscaled_y_test = unscaled_y[:ohlcv_histories.shape[0]-n]

print('ohlcv_train.shape: ', ohlcv_train.shape)
print('ohlcv_test.shape: ',ohlcv_test.shape)


# model architecture
lstm_input = Input(shape=(history_points, 5), name='lstm_input')
x = LSTM(50, name='lstm_0')(lstm_input)
x = Dropout(0.2, name='lstm_dropout_0')(x)
x = Dense(64, name='dense_0')(x)
x = Activation('sigmoid', name='sigmoid_0')(x)
x = Dense(1, name='dense_1')(x)
output = Activation('linear', name='linear_output')(x)

model = Model(inputs=lstm_input, outputs=output)
adam = tf.keras.optimizers.Adam(lr=0.0005)
model.compile(optimizer=adam, loss='mse')
model.fit(x=ohlcv_train, y=y_train, batch_size=32, epochs=500, shuffle=True, validation_split=0.1)  # 반복학습 50회.
# evaluation
y_test_predicted = model.predict(ohlcv_test)
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
y_predicted = model.predict(ohlcv_histories)
y_predicted = y_normaliser.inverse_transform(y_predicted)

assert unscaled_y_test.shape == y_test_predicted.shape
real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
print(scaled_mse)

from datetime import datetime
model.save(f'basic_model.h5')





plt.gcf().set_size_inches(22, 15, forward=True)

start = 0
end = -1

# real = plt.plot(unscaled_y_test[start:end], label='real')
# pred = plt.plot(y_test_predicted[start:end], label='predicted')

real = plt.plot(unscaled_y[start:end], label='real')
pred = plt.plot(y_predicted[start:end], label='predicted')

plt.legend(['Real', 'Predicted'])
plt.title('SK Hynix Using LSTM by TGG')
plt.show()


col_name = ['real', 'pred']
real, pred = pd.DataFrame(unscaled_y[start:end]), pd.DataFrame(y_predicted[start:end])
foo = pd.concat([real, pred], axis = 1)
foo.columns = col_name