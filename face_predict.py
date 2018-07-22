import cv2
import pickle
import numpy as np

face_cascade=cv2.CascadeClassifier("lbpcascade_frontalface.xml")
face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("trainer.yml")
labels={"person name":1}

with open("labels.pickle",'rb') as f:
    original_labels=pickle.load(f)
    labels={v:k for k,v in original_labels.items()}

img=cv2.imread(r"D:\PythonExperiments\testing\s3\9338446.20.jpg")
img1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(img1, scaleFactor=1.3, minNeighbors=5)
for (x, y, w, h) in faces:
    roi = img1[y:y + h, x:x + w]
id_,conf=face_recognizer.predict(roi)

cv2.putText(roi,labels[id_],(x,y),cv2.FONT_HERSHEY_COMPLEX,1.5,(255,255,255),2,cv2.LINE_AA)
#Put text


print(labels[id_])
cv2.imshow('img',roi)

cv2.waitKey(0)
cv2.destroyAllWindows()
