import mediapipe as mp
import cv2
import gaze
import numpy as np

mp_face_mesh = mp.solutions.face_mesh


# camera stream:
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
        max_num_faces=1,                            # number of faces to track in each frame
        refine_landmarks=True,                      # includes iris landmarks in the face mesh model
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    
    while cap.isOpened():
        success,image = cap.read()
        
        if not success:
            print("Ignoring empty camera frame")
            continue
    
        image.flags.writeable = False
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            
        cv2.imshow("output window",image)
        if cv2.waitKey(2) & 0xFF == 27:
            break
cap.release()