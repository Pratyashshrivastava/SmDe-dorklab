#importing libraries
import cv2
import numpy as np

# loading cascade
faceCascade=cv2.CascadeClassifier('face.xml')
# eye_cascade=cv2.CascadeClassifier('eye.xml')
smileCascade=cv2.CascadeClassifier('smile.xml')

# make a recognition function
# def detect(grayimage,colorimage):
#     '''cv2 only detect image in black and white format
#     and we have to convert it into color image after we 
#     complete the detection'''
#     '''
#     params: grayimage(black and white image for cv2), color image for original processing
#     '''
#     faces=face_cascade.detectMultiScale(grayimage,1.3,5)
#     '''plot a rectange on that face'''
#     for (x,y,w,h) in faces:    #face loop start here
#         cv2.rectangle(colorimage,(x,y),(x+w,y+h),(255,0,0),2)
#         roi_gray=grayimage[y:y+h,x:x+w]
#         roi_color=colorimage[y:y+h,x:x+w]
#         eyes=eye_cascade.detectMultiScale(roi_gray,1.1,22)
#         for (ex,ey,eh,ew) in eyes:     #eyes loop start here because we only have to focus inside face region
#             cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
#         smiles = smile_cascade.detectMultiScale(roi_gray,1.9,27)
#         for (sx,sy,sw,sh) in smiles:   #smile loop starts here because we only have to focus inside face region
#             cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
#     return colorimage
        
# Reading real time image 
# video_capture =cv2.VideoCapture(0)
cap=cv2.VideoCapture(0)
while(True):
    ret, img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30,30)
    )

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi_gray=gray[y:y+h, x:x+w]

        smile=smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=15,
            minSize=(25,25),
            )

        for i in smile:
            if len(smile)>1:
                cv2.putText(img,"Smiling",(x,y-30),cv2.FONT_HERSHEY_COMPLEX,
                2,(0,0,255),3,cv2.LINE_AA)
    cv2.imshow('video',img)
    k=cv2.waitKey(30) & 0xff
    if k==27: #press esc to exit
        break

cap.release()
cv2.destroyAllWindows()