import numpy as np
import cv2

cap= cv2.VideoCapture(0)
cap.set(3, 480)
cap.set(4, 480)
faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")

while True:
    success, img = cap.read()
    face = faceCascade.detectMultiScale(img, 2, 4)
    imgc = img.copy()
    imgc1 = img.copy()
    print(face)
    
    for(x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        pts1 = np.float32([[x, y], [x+w, y], [x, y+h], [x+w, y+h]])
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

        matr = cv2.getPerspectiveTransform(pts1, pts2)
        imgo = cv2.warpPerspective(imgc1, matr, (w, h))

        imgc = cv2.GaussianBlur(imgc1,(7, 7), 100000)
        imgc = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)
        imgc = cv2.cvtColor(imgc,cv2.COLOR_GRAY2RGB)
        imgc[y:y+h, x:x+h] = imgo

    imgh = np.hstack((img, imgc))
    cv2.imshow("Video2", imgh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        img1=imgc.copy()
        break


cv2.imshow("Final", img1)
cv2.waitKey(0)



