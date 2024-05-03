# 뉴럴 네트워크는 숫자만 넣어야함
# 픽셀을 숫자로 변경하여 대입(컬러: rgb(r, g, b), 흑백: 0 ~ 255 )

import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
# tensorflow에서 제공하는 데이터셋      ( (쇼핑몰이미지, 정답), (테스트용X, 테스트용Y) ) 형태
(trainX, trainY), (testX, testY) = keras.datasets.fashion_mnist.load_data()

# numpy array 자료의 shape 변경
# trainX.reshape( (60000, 28, 28, 1))
trainX = trainX.reshape( ( trainX.shape[0], 28, 28, 1) )
testX = testX.reshape( ( testX.shape[0], 28, 28, 1) )

categories = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankleboot']

# 모델만들기
# cross_entropy 라는 loss함수 사용
# relu : 음수는 다 0으로 만듬(convolution layer에서 자주사용)
# sigmoid : 결과를 0~1로 압축 - binary 예측 문제에 사용, 마지막 노드 갯수는 1개
# softmax : 결과를 0~1로 압축 - 카테고리 예측문제에 사용 - 카테고리별 확률을 다 더하면 1
# input_shape = 데이터하나의 shape 넣어주면 summary 보기 가능
model = keras.Sequential([
    # CNN 적용 ( # 32개의 다른 feature를 생성, (3, 3) => kernel 사이즈, padding - 축소된 이미지크기 보정, input_shape이 없으면 ndim 에러가 발생 )
    # Conv2D는 4차원 데이터 입력필요(ex: (60000, 28, 28, 1))
    keras.layers.Conv2D( 32, (3, 3) , padding="same", activation="relu", input_shape=(28, 28, 1)),     
    keras.layers.MaxPooling2D( (2, 2) ),        # (2, 2) => pooling 사이즈
    # keras.layers.Dense(128, input_shape=(28, 28), activation="relu"),   # 학습시엔 input_shape 없이도 알아서 판단
    keras.layers.Flatten(),                 # 행렬을 1차원으로 압축해주는 Flatten 레이어 [[1,2,3,4], [5,6,7,8]] => [1, 2, 3, 4, 5, 6, 7, 8]
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),     # 카테고리 10개 중에 특정카테고리일 확률 => 확률예측문제라면 마지막 레이어 노드수를 카테고리 갯수만큼
])

# 모델 아웃라인 출력
# model.summary()
# exit()
model.compile( loss="sparse_categorical_crossentropy", optimizer="adam", metrics="accuracy") # => 카테고리 예측문제에서 쓰는 loss
# model.fit(trainX, trainY, epochs=5 )

# score = model.evaluate(testX, testY)   # 학습 후 모델 평가하기 model.evaluate( X데이터, Y데이터) -> 컴퓨터가 처음 보는 데이터를 넣어야함
# print(score)
# overfitting 현상 - 마지막 epoch의 accuracy와 evaluate의 accuracy와 차이가 존재 - training 데이터셋을 외워서 accuracy를 높였기 때문

# validation_data(X, Y) -> epoch 1회 끝날 때마다 평가하는방법
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=5)
