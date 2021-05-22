import cv2
import numpy as np
import face_recognition

# STEP 1 : Loading the images...

imgNM = face_recognition.load_image_file('ImagesBasic/NM.jpg')
imgNM = cv2.cvtColor(imgNM, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/AB.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# STEP 2 : Detecting faces in the images...

# face locations finding process!
faceLoc = face_recognition.face_locations(imgNM)[0]

# Encoding the detected faces!
encodeNM = face_recognition.face_encodings(imgNM)[0]
# print(faceLoc)

# To encircle it with a rectangular box
cv2.rectangle(imgNM, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLoctest = face_recognition.face_locations(imgTest)[0]
encodeNMTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLoctest[3], faceLoctest[0]), (faceLoctest[1], faceLoctest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeNM], encodeNMTest)
faceDis = face_recognition.face_distance([encodeNM],encodeNMTest)
print(results, faceDis)
cv2.putText(imgTest, f'{results}, {round(faceDis[0],2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)


cv2.imshow('NAMO', imgNM)
cv2.imshow('NAMO TEST', imgTest)
cv2.waitKey(0)