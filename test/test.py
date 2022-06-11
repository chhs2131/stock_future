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