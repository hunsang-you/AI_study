# google colab 사용
# import os
# os.environ['KAGGLE_CONFIG_DIR'] = '../key/kaggle.json'
# kaggle에서 json 파일 colab에 업로드
# !kaggle competitions download -c dogs-vs-cats-redux-kernels-edition
# !unzip -q /content/dogs-vs-cats-redux-kernels-edition.zip -d .
# !unzip -q train.zip -d .

# 이미지를 변환
# 1. opencv 라이브러리로 이미지 숫자화 하기
# 2. keras 이용하여 처리(<<)

import tensorflow as tf
# import shutil
# dataset 안에 dog, cat 폴더 각각 생성
# train안의 모든 파일명에 cat이 있다면 cat폴더로, dog가 있다면 dog폴더로

# for name in os.listdir('/content/train'):
#   if 'cat' in name:
#     # (어떤 파일을, 어떤 경로로)
#     shutil.copyfile('/content/train/' + name, '/content/dataset/cat/' + name)

#   elif 'dog' in name:
#     shutil.copyfile('/content/train/' + name, '/content/dataset/dog/' + name)




# 폴더 내 이미지들을 바로 Dataset 만들어줌
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    # 로컬이라면 데이터의 위치대로 경로
    './dataset/',
    image_size = (64, 64),    # 모든 이미지가 64 * 64 픽셀
    batch_size = 64,          # 이미지 2만장을 한번에 넣지않고 batch 숫자만큼 넣음
    subset="training",
    validation_split=0.2,      # 데이터중 80%
    seed=1234
    )
# train_ds => ( (xxx...xx), (yy....yy) ) 의 형태를 train_ds 모델에 집어 넣으면 학습 끝

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    # 로컬이라면 데이터의 위치대로 경로
    './dataset/',
    image_size = (64, 64),
    batch_size = 64,
    subset="validation",
    validation_split=0.2,      # 데이터중 20%
    seed=1234
    )

print(train_ds)

# 인풋 데이터 0~1사이로 압축하기
# 전처리 함수
def 전처리함수(i, answer):
  i = tf.cast(i/255.0, tf.float32)    # 자료 타입
  return i, answer

train_ds = train_ds.map(전처리함수)
val_ds = val_ds.map(전처리함수)
# Found 25000 files belonging to 2 classes.
# Using 20000 files for training.
# Found 25000 files belonging to 2 classes.
# Using 5000 files for validation.
# train_ds는 ( (이미지 2만개), (정답 2만개) )

# import matplotlib.pyplot as plt

for i, answer in train_ds.take(1):
  print(i)
  print(answer)
#   plt.imshow(i[0].numpy().astype('uint8'))
#   plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D( 32, (3, 3) , padding="same", activation="relu", input_shape=(64, 64, 3)),   # 64 * 64 칼라사진이므로 3     
    tf.keras.layers.MaxPooling2D( (2, 2) ),      
    tf.keras.layers.Conv2D( 64, (3, 3) , padding="same", activation="relu"),   # 64 * 64 칼라사진이므로 3     
    tf.keras.layers.MaxPooling2D( (2, 2) ),   
    tf.keras.layers.Dropout(0.2),      # overfitting 완화기능: 윗레이어의 노드를 일부제거해줌   
    tf.keras.layers.Conv2D( 128, (3, 3) , padding="same", activation="relu"),   # 64 * 64 칼라사진이므로 3     
    tf.keras.layers.MaxPooling2D( (2, 2) ),      
    tf.keras.layers.Flatten(),                
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),       # 개인지 고양이 인지 0~1 사이 값이 출력되므로 1
])

model.compile( loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]) 
# train_ds ((이미지들), (정답)),  val_ds ((이미지들), (정답)) 형태
model.fit(train_ds, validation_data=val_ds, epochs=5)