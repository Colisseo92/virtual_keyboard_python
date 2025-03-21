import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

def calculate_ear(landmarks, eye_points):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_points]
    
    vertical_1 = ((p2[0] - p6[0]) ** 2 + (p2[1] - p6[1]) ** 2) ** 0.5
    vertical_2 = ((p3[0] - p5[0]) ** 2 + (p3[1] - p5[1]) ** 2) ** 0.5
    horizontal = ((p1[0] - p4[0]) ** 2 + (p1[1] - p4[1]) ** 2) ** 0.5

    return (vertical_1 + vertical_2) / (2.0 * horizontal)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

ear_history = deque(maxlen=10)  # Store last 10 EAR values for dynamic thresholding
eyes_closed = False 

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,  # Detect only one face
    refine_landmarks=False,  # Turn off refined landmarks (not needed here)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

frame_counter = 0
screen_x = 0
screen_y = 0

while True:
    success, image = cap.read()
    
    if not success:
        break
    
    start = time.time()
    
    image = cv2.cvtColor(cv2.flip(image,1),cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    results = face_mesh.process(image)
    
    image.flags.writeable = True
    
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    
    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = {i: (lm.x, lm.y) for i, lm in enumerate(face_landmarks.landmark)}

            # Convert to screen coordinates
            h, w, _ = image.shape
            landmarks = {k: (int(v[0] * w), int(v[1] * h)) for k, v in landmarks.items()}

            # Compute EAR
            left_ear = calculate_ear(landmarks, LEFT_EYE)
            right_ear = calculate_ear(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0

            # Update rolling EAR history
            ear_history.append(avg_ear)
            smoothed_ear = np.mean(ear_history)  # Moving average of last 10 frames

            # Compute EAR drop percentage
            if smoothed_ear > 0:  # Avoid division by zero
                ear_drop = (smoothed_ear - avg_ear) / smoothed_ear  # Relative drop

                # Detect sudden drop (e.g., EAR drops by 25% compared to recent average)
                if ear_drop > 0.25 and not eyes_closed:
                    print("ALERT: Sudden eye closure detected!")  # Replace with action
                    pyautogui.click()
                    eyes_closed = True  # Set flag to prevent repeated messages

                # Reset flag when eyes reopen
                if ear_drop < 0.1:  # If EAR stabilizes again
                    eyes_closed = False
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 160 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 160:
                        nose_2d = (lm.x*img_w,lm.y*img_h)
                        nose_3d = (lm.x*img_w,lm.y*img_h,lm.z*3000)
                    
                    x,y = int(lm.x*img_w),int(lm.y*img_h)
                    
                    face_2d.append([x,y])
                    face_3d.append([x,y,lm.z])
                    
            face_2d = np.array(face_2d,dtype=np.float64)
            face_3d = np.array(face_3d,dtype=np.float64)
            
            focal_length = 1*img_w
            
            cam_matrix = np.array([
                [focal_length,0,img_w/2],
                [0, focal_length, img_h/2],
                [0,0,1]
            ])
                
            dist_matrix = np.zeros((4,1), dtype=np.float64)
            
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d,face_2d,cam_matrix,dist_matrix)
            
            rmat, jac = cv2.Rodrigues(rot_vec)
            
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            
            x = angles[0] * 360
            y = angles[1] * 360

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y*10), int(nose_2d[1] - x * 10))

            norm_x = p2[0] / img_w  # Convert to range [0, 1]
            norm_y = p2[1] / img_h

            # Convert to screen coordinates
            screen_x = int(1920+norm_x * 1920)
            screen_y = int(norm_y * 1080)

            cv2.line(image,p1,p2,(255,0,0), 3)
        # Move the mouse cursor
        
        end = time.time()
        totalTime = end - start
        
        fps = 1/totalTime
        
        cv2.putText(image,f'FPS:{int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0.255,0), 2)
    pyautogui.moveTo(screen_x, screen_y, duration=0.1)
    cv2.imshow("Headpose estimation", image)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break
    

# Release camera when done
cap.release()
cv2.destroyAllWindows()