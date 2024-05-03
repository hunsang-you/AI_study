import tensorflow as tf

# 키와 신발사이즈의 관련성
# height = [170, 180, 175, 160]
# shoes = [260, 270, 265, 255]

# shoes = a * height + b
height = 170
shoes = 270

a = tf.Variable(0.1)
b = tf.Variable(0.2)

# 손실함수
def loss_function():
    predict = height * a + b
    return tf.square(260 - predict)

# 경사하강법
opt = tf.keras.optimizers.Adam(learning_rate=0.1)

for i in range(300):    # 원하는 만큼 경사하강 반복
    opt.minimize(loss_function, var_list=[a, b])      # <- 경사하강 1번 실행 == a, b를 1번 수정
    # print(a.numpy(), b.numpy())

print(170 * a.numpy() + b.numpy())