import cv2
import face_recognition
import numpy as np
import os

path = 'Images Attendence'
images = []
classnames = []
mylist = os.listdir(path)

for cl in mylist:
    curImg = cv2.imread(path+"/"+cl)
    images.append(curImg)
    classnames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

ImageEncodingsKnown = findEncodings(images)
print("Encoding complete")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodingsCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace, faceLoc in zip(encodingsCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(ImageEncodingsKnown, encodeFace)
        dist = face_recognition.face_distance(ImageEncodingsKnown, encodeFace)
        print(matches)
        print(dist)

        matchIndex = np.argmin(dist)

        if matches[matchIndex]:
            name = classnames[matchIndex].upper()
        else:
            name = "Unknown"
        y1,x2,y2,x1=faceLoc
        y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Web Cam", img)
    cv2.waitKey(1)