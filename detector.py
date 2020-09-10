import cv2
import numpy as np
import os


faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
rec=cv2.face.LBPHFaceRecognizer_create();
rec.read("recognizer\\trainingData.yml")

font = cv2.FONT_HERSHEY_SIMPLEX
fontscale=1
fontcolor=(255,255,0)

id=0
cam=cv2.VideoCapture(0);
while(True):
    ret,img=cam.read();
    gray=cv2.cvtColor (img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces :
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2);
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if(conf < 100):
            conf="{0}% ".format (round(conf))
            if(id==1):
                 id="Johnny Depp"
            elif(id==2):
                  id="Angelina Jolie"
            else:
                  id="Taninmayan kisi"
                  conf="{0}% ".format(round(conf))
        cv2.putText(img, str(id), (x,y+h), font, fontscale, fontcolor);
        cv2.putText(img, str(conf), (x+50,y), font, fontscale, fontcolor);
    cv2.imshow("Face",img);
    if(cv2.waitKey(1)== ord ('q')):
       break
    
cam.release()
cv2.destroyAllWindows()
