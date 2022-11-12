import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0
#use mediapipe library to find differfent points in face
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec =mpDraw.DrawingSpec(thickness=1,circle_radius=2,color=(255,255,255))
while True:
    success, img = cap.read()
    #Convert the image intp RGB

    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    #display
    faces = []
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,faceLms,mpFaceMesh.FACEMESH_TESSELATION,
                                 drawSpec,drawSpec )
            face = []
            for id, lm in enumerate(faceLms.landmark):
                #print(lm)
                ih, iw, ic = img.shape
                x,y,z = int(lm.x*iw),int(lm.y*ih), int(lm.z*ic)
                print(id,x,y,z)
                cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                            0.5, (0, 255, 0), 1)
                face.append([x, y])
            faces.append(face)

    if len(faces)!=0:
        print(len(faces[0]))
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,
                3,(0,255,0),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)
