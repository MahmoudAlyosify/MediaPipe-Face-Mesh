import requests
import cv2
import numpy as np
import imutils
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
faseMesh = mp_face_mesh.FaceMesh()
mp_draw=mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    result = faseMesh.process(img)
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            mp_draw.draw_landmarks(img,face_landmarks)

    print(result.multi_face_landmarks)
    cv2.imshow("Mahmoud Face Mesh", img)
    cv2.waitKey(0)
