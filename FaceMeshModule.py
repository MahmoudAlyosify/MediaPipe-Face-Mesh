import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self,staticMode=False, maxFaces = 10, minDetectionCon = 0.5, minTrackCon = 0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon



        #use mediapipe library to find differfent points in face
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,self.maxFaces,
                                                 self.minDetectionCon,self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1,circle_radius=2,color=(255,255,255))

    def findFaceMesh(self,img, draw = True):

        #Convert the image intp RGB

        self.imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
         #display
        faces = []
        if self.results.multi_face_landmarks:

            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,faceLms,self.mpFaceMesh.FACEMESH_TESSELATION,
                                 self.drawSpec,self.drawSpec )
                face =[]
                for id, lm in enumerate(faceLms.landmark):
                    #print(lm)
                    ih, iw, ic = img.shape
                    x,y,z = int(lm.x*iw),int(lm.y*ih), int(lm.z*ic)
                    cv2.putText(img, str(id), (x,y), cv2.FONT_HERSHEY_PLAIN,
                                0.7, (0, 255, 0), 1)

                    #print(id,x,y,z)
                    face.append([x,y,z])
                faces.append(face)
            return img, faces



def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector(maxFaces = 10)
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        if len(faces)!= 0:
            print(len(faces[0]))
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime


        cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
if __name__ == "__main__":
    main()