import cv2


faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);


id=input('enter user id:')

sayac=0;
while(True):
    ret,img=cam.read();
    gray=cv2.cvtColor (img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces :
        sayac=sayac+1;
        cv2.imwrite("dataSet/User."+str (id)+"."+str(sayac)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2);
        cv2.waitKey(100);
    cv2.imshow("Face",img);
    cv2.waitKey(1);
    if(sayac>20):
       break
    
cam.release()
cv2.destroyAllWindows()
