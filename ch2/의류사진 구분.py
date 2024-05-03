# 뉴럴 네트워크는 숫자만 넣어야함
# 픽셀을 숫자로 변경하여 대입(컬러: rgb(r, g, b), 흑백: 0 ~ 255 )

import tensorflow as tf
import keras
import matplotlib.pyplot as plt
# tensorflow에서 제공하는 데이터셋      ( (쇼핑몰이미지, 정답), (테스트용X, 테스트용Y) ) 형태
(trainX, trainY), (testX, testY) = keras.datasets.fashion_mnist.load_data()

categories = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankleboot']

# plt.imshow(trainX[1])
# plt.gray()
# plt.colorbar()
# plt.show()

# 모델만들기
# cross_entropy 라는 loss함수 사용
# relu : 음수는 다 0으로 만듬(convolution layer에서 자주사용)
# sigmoid : 결과를 0~1로 압축 - binary 예측 문제에 사용, 마지막 노드 갯수는 1개
# softmax : 결과를 0~1로 압축 - 카테고리 예측문제에 사용 - 카테고리별 확률을 다 더하면 1
# input_shape = 데이터하나의 shape 넣어주면 summary 보기 가능
model = keras.Sequential([
    keras.layers.Dense(128, input_shape=(28, 28), activation="relu"),   # 학습시엔 input_shape 없이도 알아서 판단
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Flatten(),                 # 행렬을 1차원으로 압축해주는 Flatten 레이어 [[1,2,3,4], [5,6,7,8]] => [1, 2, 3, 4, 5, 6, 7, 8]
    keras.layers.Dense(10, activation='softmax'),     # 카테고리 10개 중에 특정카테고리일 확률 => 확률예측문제라면 마지막 레이어 노드수를 카테고리 갯수만큼
])

# 모델 아웃라인 출력
# model.summary()
# exit()
model.compile( loss="sparse_categorical_crossentropy", optimizer="adam", metrics="accuracy") # => 카테고리 예측문제에서 쓰는 loss
model.fit(trainX, trainY, epochs=5 )
