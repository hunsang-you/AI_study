# google colab 사용
import os
os.environ['KAGGLE_CONFIG_DIR'] = '/content/'
# kaggle에서 json 파일 colab에 업로드
# !kaggle competitions download -c dogs-vs-cats-redux-kernels-edition
# !unzip -q /content/dogs-vs-cats-redux-kernels-edition.zip -d .
# !unzip -q train.zip -d .

# 이미지를 변환
# 1. opencv 라이브러리로 이미지 숫자화 하기
# 2. keras 이용하여 처리(<<)

import tensorflow as tf
import shutil
# dataset 안에 dog, cat 폴더 각각 생성
# train안의 모든 파일명에 cat이 있다면 cat폴더로, dog가 있다면 dog폴더로

for name in os.listdir('/content/train'):
  if 'cat' in name:
    # (어떤 파일을, 어떤 경로로)
    shutil.copyfile('/content/train/' + name, '/content/dataset/cat/' + name)

  elif 'dog' in name:
    shutil.copyfile('/content/train/' + name, '/content/dataset/dog/' + name)


# 폴더 내 이미지들을 바로 Dataset 만들어줌
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    # 로컬이라면 데이터의 위치대로 경로
    '/content/dataset/',
    image_size = (64, 64),    # 모든 이미지가 64 * 64 픽셀
    batch_size = 64,          # 이미지 2만장을 한번에 넣지않고 batch 숫자만큼 넣음
    subset="training",
    validation_split=0.2,      # 데이터중 80%
    seed=1234
    )
# train_ds => ( (xxx...xx), (yy....yy) ) 의 형태를 train_ds 모델에 집어 넣으면 학습 끝

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    # 로컬이라면 데이터의 위치대로 경로
    '/content/dataset/',
    image_size = (64, 64),    
    batch_size = 64,          
    subset="validation",
    validation_split=0.2,      # 데이터중 20%
    seed=1234
    )

print(train_ds) 
# Found 25000 files belonging to 2 classes.
# Using 20000 files for training.
# Found 25000 files belonging to 2 classes.
# Using 5000 files for validation.
# train_ds는 ( (이미지 2만개), (정답 2만개) )