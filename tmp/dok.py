import numpy as np
import cv2
import tensorflow as tf

from models.mnist_dok_0521 import model, checkpoint, input_shape

model.load_weights(checkpoint)

def nothing(x):
    pass

def getNumericKey(key):
    # 0~9사이의 키가 눌렸을 때만 해당 키를 반환.
    # 그 외의 키가 눌리면 None을 반환한다.
    c = chr(key)
    if c in "0123456789":
        return int(c)
    return None

def writeText(frame, txt, color=(128, 128, 255), pos=(32,16)):
    cv2.putText(frame,
                txt,
                pos,               # Coordinates
                cv2.FONT_HERSHEY_PLAIN, # 
                1.2,                    # Font scale
                color,        # Font color
                lineType=cv2.LINE_AA)


#cv2.namedWindow('Binary')
capture = cv2.VideoCapture("res/img/자동차 번호판.jpeg")
#capture = cv2.VideoCapture(0)

# --------------------------------
# mnist 데이터베이스를 이용한 학습
# --------------------------------

while True:
    ret, frame = capture.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 두 이미지에서 모두 mask값에 해당하는 부분만 저장
    ret, binary = cv2.threshold(gray_frame, 128, 255, cv2.THRESH_BINARY_INV)

    contours, hierachy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #x, y, width, height = cv2.boundingRect(contours[i])
    # connectedComponentsWithStat() : Python 3.0 부터 생긴 라벨링 함수
    # numofLabels : 레이블 개수 (반환값이 N인 경우 0~N-1까지의 레이블 번호 존재)
    # img_label : 레이블링된 입력 영상과 같은 크기의 배열
    # stats : N x 5 행렬(N:레이블 개수), [x좌표,y좌표,폭,높이,넓이]
    # centroids :  각 레이블의 중심점 좌표, N X 2 행렬
    # yellow_range : 입력 영상
    # print('')
    #numOfLabels, img_label, stats, centroids = cv2.connectedComponentsWithStats(yellow_range)

    padding = 5

    for cont in contours:
        if cv2.contourArea(cont) < 50 : continue

        x,y,w,h = cv2.boundingRect(cont)
        cv2.rectangle(frame, (x-padding, y-padding), (x+w+padding, y+h+padding), (0, 0, 255), 2)

        frame_cut = gray_frame[y-padding:y+h+padding, x-padding:x+w+padding]
        frame_cut = cv2.resize(frame_cut, (448, 448))
        frame_cut = cv2.resize(frame_cut, (28, 28))

        x_predict = tf.expand_dims(frame_cut.reshape(input_shape), axis=0)  # 수정한부분
        p = model.predict(x_predict)[0]  # 학습된 모델을 사용하여 결과 예측

        y_predict = np.where(p == max(p))[0][0]  # one-hot 인코딩에서 0~9사이의 숫자로 변환

        writeText(frame, str(y_predict), pos=(x,y), color=(32,32,32))

        # -----------------------------------------------------------
        # Key mappings
        # -----------------------------------------------------------
        key = cv2.waitKey(20) & 0xFF
        if key == 27: break  # ESC

    cv2.imshow("original", frame)
    # 이미지를 갱신하기 위해 waitKey()를 이용해 50ms만큼 대기한 후 다음 프레임으로 넘어감
    # q가 입력되면 동영상 재생 중지 -> Python OpenCV는 문자를 처리하지 못하므로 유니코드 값으로 변환하기 위해 ord() 사용
    if cv2.waitKey(0) == ord('q'): break

# 동영상 재생이 끝난 후 동영상 파일을 닫고 메모리를 해제
capture.release()
# 윈도우를 닫음
cv2.destroyAllWindows()