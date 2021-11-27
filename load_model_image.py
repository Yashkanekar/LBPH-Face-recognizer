import numpy as np
import cv2
import os

import faceRecognition as fr
print (fr)

test_img=cv2.imread(r'C:\Users\Admin\Desktop\mini project Face-Recognition-master\test images\muskma.jpg')


faces_detected,gray_img=fr.faceDetection(test_img)
print("face Detected: ",faces_detected)


face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'C:\Users\Admin\Desktop\mini project Face-Recognition-master\trainingData.yml') 

name={0:"Yash",1:"Modi",2:"Gates",3:"Jack",4:"Elon",5:"Trump"}             

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print ("Confidence :",confidence)
    print("label :",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    fr.put_text(test_img,predicted_name,x,y)

resized_img=cv2.resize(test_img,(1000,700))

cv2.imshow("face detection ", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows