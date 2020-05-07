# -----------------------------------------------------------
# 2020-05-07 세미나 발표 시연용
# -----------------------------------------------------------

import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def writeText(frame, txt):
    # 화면 프레임에 글씨 쓰기
    frame[:28,:] = frame[:28,:] // 2
    cv2.putText(frame,
                txt,
                (32, 16),               # Coordinates
                cv2.FONT_HERSHEY_PLAIN, # 
                1.2,                    # Font scale
                (128, 128, 255),        # Font color
                lineType=cv2.LINE_AA)

def getNumericKey(key):
    # 0~9사이의 키가 눌렸을 때만 해당 키를 반환.
    # 그 외의 키가 눌리면 None을 반환한다.
    c = chr(key)
    if c in "0123456789":
        return int(c)
    return None

# -----------------------------------------------------------
#
# 모델에 대한 정의 및 mnist를 이용한 학습
#
# -----------------------------------------------------------

input_shape = (28,28,1)

# --------------------------------

model = tf.keras.models.Sequential([
    Conv2D(filters=32,
           kernel_size=(3,3),
           padding='same',
           activation='relu',
           input_shape=input_shape),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),
    
    Conv2D(filters=32,
           kernel_size=(3,3),
           padding='same',
           activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(128,
          activation='relu'),
    Dropout(0.2),
    Dense(10,
          activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
# --------------------------------
# mnist 데이터베이스를 이용한 학습
# --------------------------------

if False:
    model.load_weights('models/checkpoints/mnist_2_layers')
else:
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = np.reshape(x_train / 255.0, tuple([x_train.shape[0]] + list(input_shape)))
    x_test  = np.reshape( x_test / 255.0, tuple([ x_test.shape[0]] + list(input_shape)))

    model.fit(x_train, y_train, epochs=12)

model.evaluate(x_test, y_test, verbose=2)

# -----------------------------------------------------------
# 
# 학습된 모델을 이용하여 실시간 손글씨 검출 및 예측
#
# -----------------------------------------------------------

vidIn = cv2.VideoCapture(0)

# --------------------------------
# 프레임을 정사각형모양으로 자르기 위한 계산
# --------------------------------

if not vidIn.isOpened():
    raise Exception('No Video Capture device available')

ret, frame = vidIn.read()
camHeight, camWidth = frame.shape[:2]

bounds = {'x0':0, 'y0': 0,
          'x1':0, 'y1': 0}

if camWidth < camHeight:
    deltas = camHeight-camWidth
    bounds['x0'] = 0
    bounds['x1'] = camWidth
    bounds['y0'] = deltas//2
    bounds['y1'] = deltas//2 + camWidth
else:
    deltas = camWidth-camHeight
    bounds['x0'] = 0
    bounds['x1'] = camHeight
    bounds['y0'] = deltas//2
    bounds['y1'] = deltas//2 + camHeight

# ===========================================================
# Main loop
# ===========================================================
while vidIn.isOpened():
    ret, frame = vidIn.read()
    if not ret: break

    frame = frame[bounds['x0']:bounds['x1'],
                  bounds['y0']:bounds['y1']]

    # frame = cv2.flip(frame, 0) # flip virtically (up and down)
    # frame = cv2.flip(frame, 1) # flip horizontally (left and right)

    # -----------------------------------------------------------
    # 손글씨 검출 및 정규화
    # -----------------------------------------------------------
    frame = cv2.resize(frame, (448, 448))
    
    yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    red_yuv_mask = cv2.inRange(yuv_frame,
                                ( 64,  0,128),
                                (255,128,255))

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_hsv_mask = cv2.inRange(hsv_frame,
                                (  0, 48, 48),
                                (255,255,255))
    red_mask = cv2.dilate(cv2.bitwise_and(red_yuv_mask, red_hsv_mask),
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9)),
                            iterations=3)

    normalized_mask = cv2.resize(red_mask,(28,28))

    # -----------------------------------------------------------
    # x(데이터)와 y(결과)를 결정
    # -----------------------------------------------------------s
    x = np.expand_dims(normalized_mask.reshape(input_shape), axis=0)
    p = model.predict(x)[0] # 학습된 모델을 사용하여 결과 예측

    y = np.where(p == max(p))[0][0] # one-hot 인코딩에서 0~9사이의 숫자로 변환

    # -----------------------------------------------------------
    # Key mappings
    # -----------------------------------------------------------
    key = cv2.waitKey(20) & 0xFF
    if key == 27: break # ESC

    # -----------------------------------------------------------
    # 모델의 예측 결과가 부정확 할 시, 재훈련
    # (재훈련에 사용할 레이블은 키보드의 숫자키 0~9로부터 입력받음)
    # -----------------------------------------------------------
    x_train = x
    y_train = np.expand_dims(getNumericKey(key), axis=0)

    if y_train[0] != None: # 레이블이 주어지면 학습
        model.fit(x_train, y_train, epochs=1)

    # -----------------------------------------------------------
    # Show the result on screen
    # -----------------------------------------------------------
    writeText(frame, "Predict: %d, Label : %s" % (y, str(y_train[0])))
    frame[:28,:28] = cv2.cvtColor(x[0], cv2.COLOR_GRAY2BGR)
    cv2.imshow('Camera', frame)

vidIn.release()
cv2.destroyAllWindows()

print("Successfully Closed")