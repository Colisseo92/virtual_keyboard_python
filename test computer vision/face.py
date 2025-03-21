import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import imutils

# Load YOLOv11 pose model
model = YOLO("YOLO11n-pose.pt")

# Open webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam
#sr = cv2.dnn_superres.DnnSuperResImpl_create()
#path = "FSRCNN-small_x3.pb"

#sr.readModel(path)
#sr.setModel("fsrcnn",3)

IMAGE_WIDTH = 1920
hFov = 48.5
KNOWN_WIDTH = 16

def estimated_distance(pixel_width,image_width=IMAGE_WIDTH):
    if pixel_width == 0:
        return None
    
    focal_length = (image_width/2)/np.tan(np.radians(hFov/2))
    distance = (KNOWN_WIDTH*focal_length)/pixel_width
    return distance

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO pose estimation
    results = model(frame)

    for result in results:
        kp = result.keypoints
        keypoints = result.keypoints.xy[0].cpu().numpy()  # Remove batch dimension
        boxes = result.boxes
        classes = result.names
        
        if keypoints.shape[0] > 4:  # Ensure at least nose and eyes are detected
            x_left,y_left = keypoints[3]
            x_right,y_right = keypoints[4]
            
            pixel_width = np.linalg.norm(np.array([x_right,y_right]) - np.array([x_left,y_left]))
            distance = estimated_distance(pixel_width,frame.shape[1])
            
            cv2.circle(frame, (int(x_left), int(y_left)), 5, (0, 255, 0), -1)
            cv2.circle(frame, (int(x_right), int(y_right)), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"Distance: {distance} cm", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #left_eye = keypoints[1].tolist()
            #right_eye = keypoints[2].tolist()
            #nose = keypoints[0].tolist()

            # Define bounding box around the face (adjust padding as needed)
            #x_min = int(min(left_eye[0], right_eye[0], nose[0])) - 20
            #y_min = int(min(left_eye[1], right_eye[1], nose[1])) - 20
            #x_max = int(max(left_eye[0], right_eye[0], nose[0])) + 20
            #y_max = int(max(left_eye[1], right_eye[1], nose[1])) + 20

            # Ensure coordinates are within frame bounds
            #x_min, y_min = max(x_min, 0), max(y_min, 0)
            #x_max, y_max = min(x_max, frame.shape[1]), min(y_max, frame.shape[0])

            # Crop the face
            #face_crop = frame[y_min:y_max, x_min:x_max]

            # Show cropped region
            #if face_crop.size > 0:
                #cv2.imshow("Cropped Face", face_crop)
                #upscaled = sr.upsample(face_crop)
                #cv2.imshow("upscaled",upscaled)
    
    # Show original frame with keypoints
    #cv2.imshow("Webcam Feed", frame)
    cv2.imshow("YOLOv8 Pose Distance Estimation", frame)
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        plt.close()