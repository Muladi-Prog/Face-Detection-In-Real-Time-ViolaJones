import os
import cv2 
import math
import numpy as np

import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

i=0
while True:

    ret, frame = cap.read()
    print()
    if(i==0):
        print(frame)
    i+=1
    if not ret:
        break
  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # print(cap.get(cv2.CAP_PROP_FPS))
    face_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if len(face_rect) == 1:
        for face in face_rect:
            x,y,w,h = face
            new_face = frame[y:y+h,x:x+w]
        
            image = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),thickness=2)


    cv2.imshow("Face", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    