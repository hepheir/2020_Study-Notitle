import numpy as np
import cv2

FACE_THICKNESS, FACE_COLOR = 2, (255,0,0)
EYES_THICKNESS, EYES_COLOR = 1, (0,255,0)

# OpenCV에 내장된 haarcascade를 사용. 일반적으로 opencv/data/haarcascades/ 폴더 내에 있다고 한다.
#   '~/OpenCV/opencv/data/haarcascades'
PATH = '/Users/hepheir/_git/Hello-Tensorflow/tensorflow/lib/python3.7/site-packages/cv2/data/'

def faceDetect():
    face_cascade = cv2.CascadeClassifier(PATH+'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(PATH+'haarcascade_eye.xml')

    cap = cv2.VideoCapture(0)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('face_detect.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 3)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), FACE_COLOR, FACE_THICKNESS)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), EYES_COLOR, EYES_THICKNESS)
                    
        cv2.imshow('frame', frame)
        out.write(frame)
        key = cv2.waitKey(20)
        if key == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

faceDetect()
        