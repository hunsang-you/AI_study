import tensorflow as tf

tensor1 = tf.constant([3, 4, 5])
tensor2 = tf.constant([6, 7, 8])
# print(tensor1 + tensor2)

tensor3 = tf.constant( [ [1, 2], 
                        [3, 4]])

# add(), subtract(), divide(), multiply()
# 행렬곱(dot product) matmul()
# tf.zeros() -> 0만 담긴 tensor 생성
# print(tf.add(tensor1, tensor2))

tensor4 = tf.zeros(10)  # 0이 10개 담긴 tensor 생성
tensor5 = tf.zeros( [2, 2, 3])  # 0이 3개 담긴 리스트를 2개 생성하고 그것을 2개 더 생성
print(tensor5)

print(tensor3.shape)    # tensor의 모양


#  tensor의 datatype (print시 dtype)
# tensor1 = tf.constant([3, 4, 5], tf.float32) <- datatype을 float형태로
# tf.cast() <- datatype을 변형

# weight를 저장하고 싶으면 Variable 만들기
# 새로운 w1 = w1 - a(dJ() / dw1)
w = tf.Variable(1.0)        # 초기값 1.0
print(w)
print(w.numpy())    # Variable의 값 출력
# assign() => Variable에 새로운 값 할당
w.assign(2)
print(w)