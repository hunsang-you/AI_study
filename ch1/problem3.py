import tensorflow as tf
import keras
import pandas as pd
import numpy as np
# csv 파일 읽기
data = pd.read_csv('gpascore.csv')

# 데이터 전처리
# print(data.isnull().sum())   # 빈칸이 있는 행의 갯수를 출력
# data = data.dropna()           # NaN/빈값 있는 행을 제거해줌
# data.fillna(100)            # 빈칸을 채워줌
# print(data['gre'].min())

data = data.dropna()

y_data = data['admit'].values

x_data = []

for i, rows in data.iterrows():
    x_data.append([rows['gre'], rows['gpa'], rows['rank']])

print(x_data)
# 딥러닝 모델(신경망 레이어)
model = keras.models.Sequential([
    # node의 갯수 64개, 128개, 1개 (보통 2의 제곱수 등으로)
    # activation => 활성함수 = sigmoid, tanh, relu 등등
    # 마지막 레이어는 항상 예측 결과를 뱉어야함
    keras.layers.Dense(64, activation='tanh'),     
    keras.layers.Dense(128, activation='tanh'),
    keras.layers.Dense(128, activation='tanh'),
    keras.layers.Dense(1, activation='sigmoid'),      # 마지막은 하나의 노드로, 0 ~ 1사이의 확률을 뱉고싶으면 sigmoid
 ])

# optimizer => adam, adagrad, adadelta, rmsprop, sgd 등
# binary_crossentropy => 결과가 0과 1사이의 분류/확률 문제에서 사용
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# model 학습
# model.fit(x데이터, y데이터, epochs )  => x에는 학습데이터, y는 실제정답, epochs=> 실행횟수
# 데이터는 python리스트 그대로가 아닌, numpy array, tf.tensor로 넣어야함


model.fit(np.array(x_data), np.array(y_data), epochs=1000)        # w 최적화

# 학습시킨 모델로 예측하기
predict = model.predict( [ [750, 3.70, 3], [400, 2.2, 1] ])
print(predict)