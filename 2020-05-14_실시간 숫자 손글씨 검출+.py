# -----------------------------------------------------------
# demonstrate how to classify multiple hand writings from videos
# being captured in real-time, using opencv-python
# 
# (C) 2020 Kim Dong Joo, Dongguk University, Gyeongju
# email hepheir@gmail.com
# -----------------------------------------------------------

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# 모델을 바꾸려면 여기서 수정. (레포지토리의 'models/' 디렉토리에서 선택)
from models.mnist_2_layers import model, checkpoint, input_shape

def writeText(frame, txt, pos=(32,16)):
    cv2.putText(frame,
                txt,
                pos,               # Coordinates
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

def createMask_red(frame):
    # 빨간 색상을 찾고, 빨간색 영역에 대한 마스크이미지를 생성한다.
    yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    red_yuv_mask = cv2.inRange(yuv_frame,
                                ( 64,  0,128),
                                (255,128,255))
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_hsv_mask = cv2.inRange(hsv_frame,
                                (  0, 48, 48),
                                (255,255,255))
    return cv2.bitwise_and(red_yuv_mask, red_hsv_mask)
    

if __name__ == '__main__':
    model.load_weights(checkpoint)

    vidIn = cv2.VideoCapture(1)
    # vidIn = cv2.VideoCapture('res/raw/TEST.MOV')

    # 적절한 커널 크기 결정
    ret, frame = vidIn.read()
    kernel_size = tuple([2 * (min(frame.shape[:1]) // 240) + 1] * 2) # 그냥 적절히 정한 마법의 숫자.

    # ===========================================================
    # Main loop
    # ===========================================================
    while vidIn.isOpened():
        ret, frame = vidIn.read()
        if not ret: break

        # -----------------------------------------------------------
        # Detect hand writings
        # -----------------------------------------------------------
        red_mask = createMask_red(frame)
        red_mask = cv2.morphologyEx(red_mask,
                                    cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size),
                                    iterations=3)
        contours, hierarchy = cv2.findContours(red_mask,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)

        # -----------------------------------------------------------
        # Normalize each hand writing
        # -----------------------------------------------------------
        x_data = []
        box    = []

        for num_cont in contours:
            # 각 컨투어(=숫자)에 대한 28x28 정규화 된 이미지 생성 루틴
            x,y,w,h = cv2.boundingRect(num_cont)
            frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))

            # 컨투어를 (dx,dy)만큼 평행이동하여, 원점 근처로 옮김.
            #   이후 과정에서 새로운 '정사각형' 프레임의 중앙에 위치시키기 위해, 적절히 dx, dy를 조정
            dx, dy = 0, 0
            if w > h:
                dx = -x
                dy = -y + (w-h)//2
            else:
                dx = -x + (h-w)//2
                dy = -y
            
            for point in num_cont:
                point[0,0] += dx
                point[0,1] += dy
            
            # 새로 생성한 정사각형 형태의 프레임에 삽입 : num_frame
            n = max([h,w])
            num_frame = np.zeros((n,n), dtype=np.uint8)
            cv2.drawContours(num_frame, [num_cont], 0, 255, -1)

            # input_shape의 형태에 맞게 맞추어주기. (28x28x1)
            _ksize = tuple([2*(n//56)+1] * 2) # 홀수를 유지, 숫자 프레임의 너비 비율에 맞게 적절한 커널 사이즈 지정
            num_frame = cv2.dilate(num_frame, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, _ksize))
            num_frame = cv2.resize(num_frame, (28, 28))
            num_frame = num_frame.reshape(input_shape)

            x_data.append(num_frame)
            box.append([x,y,w,h])

        # -----------------------------------------------------------
        # Predict using pre-trained model
        # -----------------------------------------------------------
        x_data = np.array(x_data)

        if len(x_data):
            # 모든 숫자에 대해 한 번에 예측
            p = model.predict(x_data)
            y_data = [np.where(_p == max(_p))[0][0] for _p in p]

            for i in range(len(x_data)):
                # 예측된 값의 확인을 용이하게 하기위해, 각 숫자 바운딩 박스 주변에 예측된 값을 출력
                x,y,w,h = box[i]
                writeText(frame, str(y_data[i]), pos=(x,y - 4))

        # -----------------------------------------------------------
        # Key mappings
        # -----------------------------------------------------------
        key = cv2.waitKey(20) & 0xFF
        if key == 27: break # ESC

        # -----------------------------------------------------------
        # Re-train model (if the prediction is wrong)
        # -----------------------------------------------------------
        # # TODO: 재학습 과정은 나중에 생각.
        # x_train = x
        # y_train = np.expand_dims(getNumericKey(key), axis=0)

        # if y_train[0] != None: # 레이블이 주어지면 학습
        #     model.fit(x_train, y_train, epochs=1)

        # -----------------------------------------------------------
        # Show the result on screen
        # -----------------------------------------------------------
        cv2.imshow('Camera', frame)

    vidIn.release()
    cv2.destroyAllWindows()

print("")
print("Successfully Closed")