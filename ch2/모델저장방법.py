# 학습시킨 모델을 저장하는법
# 저장되는 항목 ( 레이어 설정, loss함수 종류, optimizer 종류, 훈련 후의 w값(가중치))
import tensorflow as tf
import keras
import numpy as np
# 의류구분 코드 이용
(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

trainX = trainX / 255.0
testX = testX / 255.0

trainX = trainX.reshape( ( trainX.shape[0], 28, 28, 1) )
testX = testX.reshape( ( testX.shape[0], 28, 28, 1) )


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)),              
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),   
])

model.summary()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=3)
# ---------------------------------
# 1-1. 전체모델 저장하기
# model.save('새폴더/model1')    # model에 정보다 담겨있음

# 1-2. 모델 불러오기
# 불러온모델 = tf.keras.models.load_model('새폴더/model1')
# 불러온모델.summary()

# 불러온 모델을 test를 넣어서 평가
# 불러온모델.evaluate(testX, testY) # => 과거에 학습했던 결과 그대로 나옴
# ---------------------------------
# 2-1. w(가중치)값만 저장/로드    (checkpoint 저장)
# epoch 중간중간에 checkpoint를 저장

# 콜백 = tf.keras.callbacks.ModelCheckpoint(
#     filepath = 'checkpoint/mnist{epoch}',     # 폴더명 작명
#     save_weights_only = True,
#     # monior='val_acc',     # val_acc가 최대가 되는 checkpoint만 저장
#     # mode = 'max',
#     save_freq = 'epoch'        # epoch 하나 끝날때마다 저장
# )

# model.fit(trainX, trainY, validation_data=(testX, testY), epochs=3, callbacks=[콜백])

# 2-2. w값 로드 및 이용
# w값만 저장해놨으면 모델을 만들고 w값(checkpoint파일)을 로드하면 됨
# model2 = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28, 1)),              
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(10, activation='softmax'),   
# ])

# model2.summary()
# model2.compile( loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model2.load_weights('checkpoint/mnist{epochs}')        # 이경로에 있는 w값을 로드
# model2.evaluate(testX, testY)