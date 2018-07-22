import numpy as np
from PIL import Image
import cv2
import pickle
import os
base_dir= os.path.dirname(os.path.abspath(__file__))
image_dir=os.path.join(base_dir,"training")

face_cascade=cv2.CascadeClassifier("lbpcascade_frontalface.xml")
face_recognizer=cv2.face.LBPHFaceRecognizer_create()

current_id=0
label_ids={}
y_labels=[]
x_train=[]

for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith("jpg") or file.endswith("png"):
            path=os.path.join(root,file)
            label=os.path.basename(root).replace(" ","-").lower()
            #print(label,path)
            if not label in label_ids:
                label_ids[label]=current_id
                current_id+=1
            id_=label_ids[label]
            #print(label_ids)
            pil_image=Image.open(path).convert("L")
            image_array=np.array(pil_image,"uint8")
            faces=face_cascade.detectMultiScale(image_array,scaleFactor=1.3, minNeighbors=5)
            for(x,y,w,h) in faces:
                roi=image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)


with open("labels.pickle",'wb') as f:
    pickle.dump(label_ids,f)

face_recognizer.train(x_train,np.array(y_labels))
face_recognizer.save("trainer.yml")





