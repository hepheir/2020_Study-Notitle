import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    key = cv2.waitKey(10)
    if key == 27: break

    cv2.imshow('cam', frame)

cv2.destroyAllWindows()
cap.release()