# ๐์ธ๊ณต์ง๋ฅ ์ฃผ๊ฐ์์ธก
Keras LSTM์ ์ด์ฉํ ์ฃผ๊ฐ์์ธก


**๋ฒ์  ์ ๋ณด**
- Python 3.7.13
- TensorFlow 2.8.2
- finance-datareader

<br/>

**๋ชจ๋ ์์กด์ฑ**
```python
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
```








<br/>
<br/>
<br/>

# ๐ข๊ฐ์
## ๋ชฉ์ฐจ
1. ๋ชฉํ
2. LSTM ๊ฐ๋
3. ๋ฐ์ดํฐ ์์ง
4. ๋ชจ๋ธ ์์ฑ ๋ฐ ํ์ต
5. ์์ธก ์งํ
6. ๊ฐ์  ๋ฐฉํฅ

<br/>

## ๋ชฉํ
![](๊ด๋ จ์๋ฃ/ppt_image/์ฌ๋ผ์ด๋3.JPG)
์ด์  50๊ฑฐ๋์ผ ๊ธฐ๋ก์ ํตํด ๋ด์ผ์ ์๊ฐ๋ฅผ ์์ธก 

<br/>

## LSTM ๊ฐ๋
![](๊ด๋ จ์๋ฃ/ppt_image/์ฌ๋ผ์ด๋4.JPG)
- RNN์ ๊ธฐ๋ฐํ ๋ฅ๋ฌ๋ ์ํคํ์ณ
- ์ด์  ๋ฐ์ดํฐ๋ฅผ ๊ฐ์ง๊ณ  ์ดํ์ ๋ฐ์ดํฐ๋ฅผ ์์ธกํ๋ ์ธ๊ณต์ง๋ฅ ๋ชจ๋ธ
- ์๋ก ์ฐ๊ด๋๋ฉฐ ์ฐ์์ ์ธ ๋ฐ์ดํฐ ํํ์ ์ ํฉ
- RNN์ด ๋จ๊ธฐ๋ฉ๋ชจ๋ฆฌ๋ง ๊ฐ์ง๊ณ  ์๋ ๊ฒ์ ๋ฐํด, LSTM์ ๋จ๊ธฐ, ์ฅ๊ธฐ ๋ฉ๋ชจ๋ฆฌ๋ฅผ ๊ฐ์ง๊ณ  ์์.
- ์ฌ์ฉ์: ํ์ฒด ์ธ์, ๋ํ ์ธ์,ย ์ด๋ธ๋ฉ๋ฆฌย ๋ํ์, Timeย Seriesย Data ๋ฑ










<br/>
<br/>
<br/>

# ๐  ๋ฐ์ดํฐ ์์ง
## ๋ฐ์ดํฐ ์์ง ๊ณผ์ 
![](๊ด๋ จ์๋ฃ/ppt_image/์ฌ๋ผ์ด๋6.JPG)

<br/>

## ์์งํ  ๋ฐ์ดํฐ
![](๊ด๋ จ์๋ฃ/ppt_image/์ฌ๋ผ์ด๋5.JPG)
- OHLCV : ์๊ฐ(open), ๊ณ ๊ฐ(high), ์ ๊ฐ(low), ์ข๊ฐ(close), ๊ฑฐ๋๋(volume)
- ๋์ค๋ฅ ์ง์(ixic_close)
- ํด๋น์ข๋ชฉ์ ๋ฐ์๋ ๋ด์ค ๊ฐ์(pagenum)

### ์์ง๋ ๋ฐ์ดํฐ ์
![](๊ด๋ จ์๋ฃ/์ฃผ๊ฐ์์ธก๋ฐ์ดํฐ์.png)


## ๋ฐ์ดํฐ ์์ง ์ฝ๋
- call_dataset ํจ์ ๋ด๋ถ์์ ๋ฐ์ดํฐ ์์ง์ ์ฒ๋ฆฌํ๊ณ  ๊ฒฐ๊ณผ๋ฅผ ๋ฐํํด์ค๋ค.
- NaverNewsCrawling ์ ์์ฒด ์ ์ํ ํฌ๋กค๋ง ๋ชจ๋์ด๋ค. (๋ณธ ๊ธ์ data_collection > news ๋๋ ํ ๋ฆฌ์ ์์นํจ)

```python
def call_dataset(stock_name, stt='2020-01-01', end='2020-02-01', history_points=50):
    import data_collection.news.NaverNewsCrawling as NaverNewsCrawling

    # ์ด๋ฆ์ผ๋ก ์ข๋ชฉ ์ ๋ณด ๊ฐ์ ธ์ค๊ธฐ
    df_krx = fdr.StockListing('KRX')
    list_krx = {}
    for i in range(len(df_krx.Name)):
        list_krx[df_krx['Name'][i]] = {"code": df_krx['Symbol'][i], "sector": df_krx['Sector'][i]}

    ticker = list_krx[stock_name]["code"]
    print(stock_name, "์ ์ข๋ชฉ์ฝ๋๋ ", ticker)

    # OHLCV ๊ฐ ๊ฐ์ ธ์ค๊ธฐ
    ohlcv = fdr.DataReader(ticker, stt, end)
    ohlcv = ohlcv.iloc[:, 0:-1]
    # ohlcv = ohlcv.loc[:,['Open','Close','Volume']]  # ์๊ฐ, ์ข๊ฐ, ๊ฑฐ๋๋๋ง ๊ฐ์ ธ์ค๊ธฐ
    # print('ohlcv: ', ohlcv)

    # ๋์ค๋ฅ ์ง์ (์ข๊ฐ) ๊ฐ์ ธ์ค๊ธฐ
    nasdaq = fdr.DataReader('IXIC', stt, end)
    nasdaq = nasdaq.loc[:, ['Close']]
    nasdaq.rename(columns={"Close": "IXIC_Close"}, inplace=True)  # ์ปฌ๋ผ๋ช ๋ฐ๊พธ๊ธฐ
    # print('nasdaq: ', nasdaq)

    # ๋ด์ค ๊ฐฏ์ ๊ฐ์ ธ์ค๊ธฐ
    nnc = NaverNewsCrawling()
    news_amount = nnc.get_news_amount_everyday_df(stock_name, stt, end)
    # print(news_amount)

    # ์ผ์๋ณ๋ก ๋ฐ์ดํฐ ํฉ์น๊ธฐ
    data_result = ohlcv.join(nasdaq).join(news_amount)
    data_result = data_result.fillna(method='ffill')  # ๊ฐ(NaN)์ด ์๋๊ฒฝ์ฐ ์ด์ ์ผ์์ ๋ฐ์ดํฐ๋ฅผ ๊ฐ์ ธ์ค๋๋กํจ.
    print(data_result)
    data_result = data_result.values

    # ๋ฐ์ดํฐ๋ฅผ 0~1 ๋ฒ์๋ก ์ค์ผ์ผ๋ง
    data_normalizer = preprocessing.MinMaxScaler()
    data_normalized = data_normalizer.fit_transform(data_result)

    # ์ดํด๋ณผ ์ผ์(window size = history_points) ๋จ์๋ก ๋ฐ์ดํฐ๋ฅผ ์ ์ฅํ๋ค.
    ohlcv_histories_normalized = np.array(
        [data_normalized[i:i + history_points].copy() for i in range(len(data_normalized) - history_points)])

    # y_data (์์ด๊ฐ)
    next_day_open_values_normalized = np.array(
        [data_normalized[:, 0][i + history_points].copy() for i in range(len(data_normalized) - history_points)])
    next_day_open_values_normalized = np.expand_dims(next_day_open_values_normalized, -1)  # 1XN ๋ฒกํฐ -> NX1 ๋ฒกํฐ๋ก

    next_day_open_values = np.array(
        [data_result[:, 0][i + history_points].copy() for i in range(len(data_result) - history_points)])
    next_day_open_values = np.expand_dims(next_day_open_values, -1)  # 1XN ๋ฒกํฐ -> NX1 ๋ฒกํฐ๋ก

    # ์ข๊ฐ ๋ฐ์ดํฐ ๊ธฐ๋ก
    close_values = np.array(
        [data_result[:, 1][i + history_points].copy() for i in range(len(data_result) - history_points)])
    close_values = np.expand_dims(close_values, -1)  # 1XN ๋ฒกํฐ -> NX1 ๋ฒกํฐ๋ก

    # ๋ฐ์ดํฐ๋ฅผ 0~1 ๋ฒ์๋ก ์ค์ผ์ผ๋ง
    y_normalizer = preprocessing.MinMaxScaler()
    y_normalizer.fit(next_day_open_values)

    # ๊ฐ ๋ฐํ
    return ohlcv_histories_normalized, next_day_open_values_normalized, next_day_open_values, y_normalizer, close_values
```









<br/>
<br/>
<br/>

# ๐ฐ ๋ชจ๋ธ

## ์ปจ์
![](๊ด๋ จ์๋ฃ/ppt_image/์ฌ๋ผ์ด๋9.JPG)
์ค๋์ ํฌํจํ 50์ผ๊ฐ์ ๋ฐ์ดํฐ๋ฅผ ๋ฐํ์ผ๋ก ๋ด์ผ์ ์๊ฐ(Open Price)๋ฅผ ์์ธกํ๋ ์ปจ์์ ๋ชจ๋ธ์ด๋ค.

<br/>

## ํ์ต ํ๊ฒฝ
![](๊ด๋ จ์๋ฃ/ppt_image/์ฌ๋ผ์ด๋11.JPG)
```python
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
model.fit(x=ohlcv_train, y=y_train, batch_size=32, epochs=2000, shuffle=True, validation_split=0.1)  # ๋ฐ๋ณตํ์ต ํ์ = epochs.
  ```
	
- ์๋ ฅ๋ฐ์ดํฐ์ ๋ํ 0~1 MinMax Scale ์ ์ฒ๋ฆฌ
- 64 hidden-layer
- Optimizer Adam ์ฌ์ฉ
- Dropout 20%
- ๋ฐ๋ณต ํ์ต ํ์ 2000ํ

### ํ์ต ์งํ ์
![img_2.png](๊ด๋ จ์๋ฃ/img_2.png)








<br/>
<br/>
<br/>

# ๐โโ๏ธ์์ธก ์งํ
![](๊ด๋ จ์๋ฃ/ppt_image/์ฌ๋ผ์ด๋13.JPG)

```python
plt.gcf().set_size_inches(22, 15, forward=True)

start = 0
end = -1
real = plt.plot(unscaled_y[start:end], label='real')
pred = plt.plot(y_predicted[start:end], label='predicted')

plt.legend(['Real', 'Predicted'])
plt.title(stock_name + ' Using LSTM by TGG')
plt.show()
```

![](๊ด๋ จ์๋ฃ/ppt_image/์ฌ๋ผ์ด๋14.JPG)










<br/>
<br/>
<br/>

# ์์ธก ๊ฒฐ๊ณผ ๋ฐ ๊ฐ์  ์ฌํญ

## ์์ธก ๊ฒฐ๊ณผ
![](๊ด๋ จ์๋ฃ/ppt_image/์ฌ๋ผ์ด๋15.JPG)

์์ธก ๋ฐ์ดํฐ๋ฅผ ๋ถ์ํด๋ณด๋ฉด,
์์์ ๋ดค๋ฏ ์ ํฉ๋๋ 99% ์ด์์ด๋ฉฐ, ์์ธก ๊ทธ๋ํ๋ฅผ ๋ณด๋ฉด ์ถ์ธ์ ๋ํ ์์ธก์ด ์ด๋์ ๋ ์ณ์ ๊ฒ์ผ๋ก ๋ณด์ธ๋ค.
ํ์ง๋ง ํฌ์ํ  ๋ ์ค์ํ ๊ฒ์ ๋ฑ๋ฝ ์์ธก์ ์ ํํ ํ๋์ง ์ด๋ค.
โ๋ฑ๋ฝโ์ ๋ง์ถ๋ค๋ ๊ฒ์ ์ฝ๊ฒ๋งํด โ์์ต์ ๋ผ ์ ์๋ค.โ๋ก ํด์ ํ  ์ ์๋๋ฐ ๋ฑ๋ฝ์ ๋ง์ถ์ง ๋ชปํ ๊ฒฝ์ฐ๊ฐ ์ ์ฒด์ ์ฝ 30% ์ ๋์ด๋ค.
์ฆ, ์ค์  ํฌ์์ ๋ฐ๋ก ์ ์ฉํ๊ธฐ๋ ์ด๋ ค์.

<br/>

## ๊ฐ์  ๋ฐฉํฅ
![](๊ด๋ จ์๋ฃ/ppt_image/์ฌ๋ผ์ด๋16.JPG)
- ํ์ต์ ์ฌ์ฉ๋๋ ์๋ ฅ ๋ฐ์ดํฐ์ ๋ํ ๊ณ ๋ฏผ์ด ํ์ํจ.
  - OHLCV๋ง์ผ๋ก ์์ธกํ๋ ๊ฒ์ด ์คํ๋ ค ๋ ์ข์ ๊ฒฐ๊ณผ๋ฅผ ๋ณด์ฌ์ค.
- ์ค์  ์ฃผ๊ฐ๋ฅผ ๋ง์ถ๋ ๊ฒ๋ณด๋ค๋ (์ฆ, ์ค์ฐจ๋ฅผ ์ค์ด๋๊ฒ๋ณด๋จ)
- ๋ฑ๋ฝ์ ์ฌ๋ถ ์์ฒด๋ฅผ ๋ง์ถ๋ ๊ฒ์ด ์คํ๋ ค ํฌ์์ ๋์์ด ๋จ. (๋ฐฉํฅ๋ง ๋ง์ถ๋ ๊ฒ)
- ์ฆ, ์ค์ฐจ๋ฅผ ์ค์ด๋ ๊ฒ ๋ณด๋จ, ๋ฑ๋ฝ์ ๋ง์ถ๋ ๊ฒ์ด ์ข๋ค. (์ค์ฐจ ๋ณด๋ค๋ ๋ฐฉํฅ์ ๋ฐฐ์ ์ ๋ ํฌ๊ฒ ๋์ด์ผํจ)
- ์์ ๊ฐ์ ์ด์ ๋ก y๋ฅผ ๋ฑ๋ฝ๋ฅ  max(0.5)์  + ์ค์ฐจmax(0.5์ ) = 1๋ก ๊ตฌ์ฑํ๋ ์์ผ๋ก ํ๋ฉด ์ข์ ๊ฒ ๊ฐ์. (์๊ฒฌ ^^)
  - ์ฆ y = ๋ฑ๋ฝ(0.5์  ์์น0.5, ํ๋ฝ 0์ ์ ๋ฐฐ์ ?) + ๋ค์๋  ์๊ฐ(max 0.5 scale)

- ๋์ ์ ํฉ๋๋ฅผ ๋ณด์์ผ๋, ์ค์ฐจ๋ณด๋ค๋ ๋ฑ๋ฝ๋ฅ  ํ๋จ์ ์ค์ ์ ๋ฌ์ผ ํ๋ค.
  - A๋ชจ๋ธ: ์์ธก ์ค์ฐจ๊ฐ ์์ ๋ชจ๋ธ
    - Ex)ย ์ฃผ์์ 100์ ์ผ, A๋ชจ๋ธ์ ๋ด์ผ ์๊ฐ๋ฅผ 99๋ก ์์ธก, ํ์ง๋ง ์ค์  ์๊ฐ๋ 101 (์ค์ฐจ๊ฐ 2์ง๋ง, ์ํด๋ฐ์)
  - B๋ชจ๋ธ: ๋์๊ฒ ์ด๋์ ์ฃผ๋ ๋ชจ๋ธย orย ์ํด๋ฅผ ๋๋ ๋ชจ๋ธ
    - Ex)ย ์ฃผ์์ 100์ ์ผ, B๋ชจ๋ธ์ ๋ด์ผ ์๊ฐ๋ฅผ 101๋ก ์์ธก, ํ์ง๋ง ์ค์  ์๊ฐ๋ 110 (์ค์ฐจ๊ฐ 9์ง๋ง, ์ด์ต๋ฐ์)

- ๋ด์ผ์ ๊ฐ๊ฒฉ์ ์ด์  50๊ฑฐ๋์ผ์ OHLCV, ๋์ค๋ฅ, ๋ด์ค๊ฐฏ์ ์ธ์ ๋ฌด์์ ์ํฅ์ ๋ฐ๋๊ฐ? 
  - ์ ๋ ์ ์ค๊ตญ ํญ์, ๋นํธ์ฝ์ธ ์ฆ์ ์ํฉ, ๋ฏธ๊ตญ ๊ตญ์ฑ ๊ธ๋ฆฌ๋?
  - IMF, ๋ฆฌ๋จผ๋ธ๋ผ๋์ค, ์ค์ผ์ผํฌ, ์ฝ๋ก๋ ์ผํฌ ๋ฑ ์ฃผ์ ์ธ์ ์ผ๋ก ๋ฐ์๋๋ ์ด๋ฒคํธ๋ค์ ๋ฐ์ํ  ๋ฐฉ๋ฒ.

- ๋์ค๋ฅ์ ๋ฃ์๋๋ ๋์ค๋ฅ ๊ฐ์ ์ถ์ขํ๋ฉฐ ์ค์ฐจ ๋ฐ์
  - ์ฐ๊ด ๊ด๊ณ๋ฅผ ํ์ํด์ ํ์ ์๋ ๊ฐ์ ์ ๊ฑฐํ๋ ๊ฒ๋ ํ๋์ ๋ฐฉ๋ฒ์ด ๋  ๋ฏ.














<br/>
<br/>
<br/>


# ๐๊ฒฐ๊ณผ ์์ธ
## ๊ทธ๋ํ
(SKํ์ด๋์ค 2020๋1์~2022๋6์)
![img_8.png](๊ด๋ จ์๋ฃ/img_8.png)
![img_7.png](๊ด๋ จ์๋ฃ/img_7.png)
![img_6.png](๊ด๋ จ์๋ฃ/img_6.png)
![img_5.png](๊ด๋ จ์๋ฃ/img_5.png)
![img_4.png](๊ด๋ จ์๋ฃ/img_4.png)
![img_3.png](๊ด๋ จ์๋ฃ/img_3.png)
![img_1.png](๊ด๋ จ์๋ฃ/img_1.png)
![img.png](๊ด๋ จ์๋ฃ/img.png)















<br/>
<br/>
<br/>

# ๐์ฐธ๊ณ ์๋ฃ

## ๋งํฌ
[(์นผ๋ผ) ๋ฅ๋ฌ๋ ์ด๋ณด๋ค์ด ํํํ๋ ์ค์ : ์ฃผ์๊ฐ๊ฒฉ ์์ธก AI](https://codingapple.com/unit/deep-learning-stock-price-ai/)

[์๊ณ์ด ์์ธก: LSTM ๋ชจ๋ธ๋ก ์ฃผ๊ฐ ์์ธกํ๊ธฐ](https://insightcampus.co.kr/2021/11/11/%EC%8B%9C%EA%B3%84%EC%97%B4-%EC%98%88%EC%B8%A1-lstm-%EB%AA%A8%EB%8D%B8%EB%A1%9C-%EC%A3%BC%EA%B0%80-%EC%98%88%EC%B8%A1%ED%95%98%EA%B8%B0/)

[์ ํ ํฌ ํ์ด์ฌ](https://wikidocs.net/book/1173)

[๊ธฐ๊ณํ์ต: ์ฃผ๊ฐ ์์ธก(์ ํํ๊ท, ์ ๊ฒฝ๋ง) ์๋](https://intrepidgeeks.com/tutorial/machine-learning-stock-price-prediction-linear-regression-neural-network-attempt)

[https://github.com/FinanceData/FinanceDataReader](https://github.com/FinanceData/FinanceDataReader)

[Stock-Prediction-Models/tesla-study.ipynb at master ยท huseinzol05/Stock-Prediction-Models](https://github.com/huseinzol05/Stock-Prediction-Models/blob/master/misc/tesla-study.ipynb)

[๋ฅ๋ฌ๋ ์ ํ๋ ๋์ด๊ธฐ](https://m.blog.naver.com/nanotoly/221497821754)

[์ ํ ํฌ ํ์ด์ฌ](https://wikidocs.net/36033)

[๏ปฟ2. Keras LSTM ์ ํ ์ ๋ฆฌ (2/5) - ๋จ์ธต-๋จ๋ฐฉํฅ & many-to-many ์ ํ](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=chunjein&logNo=221589624838)

### RNN LSTM ์ฃผ๊ฐ ์์ธกํ๊ธฐ

์ค์ตํ๋ฉด์ ์งํํ ์ฝ๋ (RNN, LSTM ๋๋ค์์)

[[ํ์ํ๋ก์ฐ2] RNN๊ณผ LSTM์ ์ด์ฉํ ์ฃผ์๊ฐ๊ฒฉ ์์ธก ์๊ณ ๋ฆฌ์ฆ ์ฝ๋](https://diane-space.tistory.com/285)

RNN๊ณผ LSTM์ ๋น๊ตํ๋ฉด์ ์ด๋ก ๊ณผ ํจ๊ป ์ค๋ช (์ฒ์๋ณด๊ธฐ๋ ๋ณต์กํ ์ ์์)

[Tensorflow 2.0 Tutorial ch7.1 - RNN ์ด๋ก  (1)](https://dschloe.github.io/python/tensorflow2.0/ch7_1_2_rnn_theory1/)

RNN LSTM ์ฃผ๊ฐ์์ธก ๊ฐ๋์ ์ผ๋ก ์ฝ๊ฒ ์์ฃผ ์ ํ์ด๋ (์์์ฝ๋ ์์)

[[LSTM/GRU] ์ฃผ์๊ฐ๊ฒฉ ์์ธก ๋ชจ๋ธ ๊ตฌํํ๊ธฐ](https://data-analysis-expertise.tistory.com/67)

[Google Colaboratory](https://colab.research.google.com/github/teddylee777/machine-learning/blob/master/04-TensorFlow2.0/01-%EC%82%BC%EC%84%B1%EC%A0%84%EC%9E%90-%EC%A3%BC%EA%B0%80%EC%98%88%EC%B8%A1/02-LSTM-stock-forecasting-with-LSTM-financedatareader.ipynb#scrollTo=xShll_EX0l8T)

[KEKOxTutorial/22_Keras๋ฅผ ํ์ฉํ ์ฃผ์ ๊ฐ๊ฒฉ ์์ธก.md at master ยท KerasKorea/KEKOxTutorial](https://github.com/KerasKorea/KEKOxTutorial/blob/master/22_Keras%EB%A5%BC%20%ED%99%9C%EC%9A%A9%ED%95%9C%20%EC%A3%BC%EC%8B%9D%20%EA%B0%80%EA%B2%A9%20%EC%98%88%EC%B8%A1.md)

### RNN LSTM ๊ธฐ๋ณธ ์์ 

์์ง์ด๋ ๊ทธ๋ฆผ์ด ์๋ ์์ 

[[์ผ๋ผ์ค] ๋ฌด์์  ํํ ๋ฆฌ์ผ 11 - LSTM(feat. RNN) ๊ตฌํํ๊ธฐ](https://ebbnflow.tistory.com/135)

๊ธ๋ก ๋์ด์์ผ๋ LSTM ์๋ ฅ๋ฐ์ดํฐ ํ์ ๊ฐ๋์ ์ฑ๊ธฐ๋๋ฐ ๋์์ด ๋จ (๊ธ์งง์)

[Keras LSTM ์๋ ฅ ํฌ๋งท์ ์ดํด Understanding Input shapes in LSTM | Keras](https://swlock.blogspot.com/2019/04/keras-lstm-understanding-input-and.html)

LSTM ๋์ ๊ตฌ์กฐ์ ์์ด์ ์ธ ํํ

[[๋จธ์ ๋ฌ๋ ์ํ๋ง] LSTM์ ๋ชจ๋  ๊ฒ](https://box-world.tistory.com/73)

์๋ ฅ ๋ฐ์ดํฐ ์ ๊ทํ ํด์ผํ๋ ์ด์ 

[Neural Network ์ ์ฉ ์ ์ Input data๋ฅผ Normalize ํด์ผ ํ๋ ์ด์ ](https://goodtogreate.tistory.com/entry/Neural-Network-%EC%A0%81%EC%9A%A9-%EC%A0%84%EC%97%90-Input-data%EB%A5%BC-Normalize-%ED%95%B4%EC%95%BC-%ED%95%98%EB%8A%94-%EC%9D%B4%EC%9C%A0)

[์ฃผ๊ฐ ์์ธก ๋ฅ ๋ฌ๋์ ์ํ ์๋ฃ๋ค](https://smecsm.tistory.com/54)

### ํ๋ค์ค ๋ํ์ด ๋ฐ์ดํฐํ๋ ์

[[Pandas] loc[ ] ๋ก ํ, ์ด ์กฐํํ๊ธฐ](https://m.blog.naver.com/wideeyed/221964700554)

[Pandas - DataFrame ์ปฌ๋ผ๋ช, ์ธ๋ฑ์ค๋ช ๋ณ๊ฒฝ](https://blog.naver.com/PostView.nhn?blogId=rising_n_falling&logNo=222061033231)

[Pandas DataFrame ์ธ๋ฑ์ค ์ด๋ฆ ๊ฐ์ ธ ์ค๊ธฐ ๋ฐ ์ค์ ](https://www.delftstack.com/ko/howto/python-pandas/pandas-get-and-set-index-name/)

[[Pandas] loc[ ] ๋ก ํ, ์ด ์กฐํํ๊ธฐ](https://m.blog.naver.com/wideeyed/221964700554)

[11. pandas DataFrame ์ธ๋ฑ์ฑ(์ด / ํ / boolean ์ธ๋ฑ์ฑ)](https://nittaku.tistory.com/111)

[Python DataFrame ์ปฌ๋ผ๋ช ์ธ๋ฑ์ค๋ก ์ค์ ํ๊ธฐ set_index](https://ponyozzang.tistory.com/616)

[[Pandas ๊ธฐ์ด] ๋ฐ์ดํฐํ๋ ์ ํฉ์น๊ธฐ(merge, join, concat)](https://yganalyst.github.io/data_handling/Pd_12/#1-%EB%8D%B0%EC%9D%B4%ED%84%B0-%ED%94%84%EB%A0%88%EC%9E%84-%EB%B6%99%EC%9D%B4%EA%B8%B0--pdconcat)

[[Python pandas] ๊ฒฐ์ธก๊ฐ ์ฑ์ฐ๊ธฐ, ๊ฒฐ์ธก๊ฐ ๋์ฒดํ๊ธฐ, ๊ฒฐ์ธก๊ฐ ์ฒ๋ฆฌ (filling missing value, imputation of missing values) : df.fillna()](https://rfriend.tistory.com/262)

## ๋ผ๋ฌธ

[LSTM ๊ธฐ๋ฐ ๊ฐ์ฑ๋ถ์์ ์ด์ฉํ ๋นํธ์ฝ์ธ ๊ฐ๊ฒฉ ๋ฑ๋ฝ ์์ธก](https://www.dbpia.co.kr/Journal/articleDetail?nodeId=NODE10501339)

[[๋ผ๋ฌธ]์ฃผ์์์ธ ์์ธก์ ์ํ ๋ฅ๋ฌ๋ ์ต์ ํ ๋ฐฉ๋ฒ ์ฐ๊ตฌ](https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=DIKO0014699312&dbt=DIKO#)
