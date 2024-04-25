import tensorflow as tf

# 데이터가 많은 경우
train_x = [1, 2, 3, 4, 5, 6, 7]
train_y = [3, 5, 7, 9, 11, 13, 15]


# 예측 모델 만들기
a = tf.Variable(0.1)
b = tf.Variable(0.1)

# predict_y = train_x * a + b

opt = tf.keras.optimizers.Adam(learning_rate=0.01)

# 손실함수 만들어 넣기(mean squared error, cross entropy 등등)
def loss_function(a, b):
    predict_y = train_x * a + b
    # tf.keras.losses.mse(실제값, 예측값)
    return tf.keras.losses.mse(train_y, predict_y)

for i in range(2900):
    # lambda:함수() -> 익명함수 만들기 ==> 함수안에 값을 넣고 싶을때
    opt.minimize(lambda:loss_function(a, b), var_list=[a, b])
    print(a.numpy(), b.numpy())