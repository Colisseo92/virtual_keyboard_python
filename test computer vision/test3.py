import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Open webcam
cap = cv2.VideoCapture(0)

# Landmark indices for eyelids and iris
LEFT_EYE_LANDMARKS = [263, 362, 386, 374, 387, 388, 390, 466, 249, 390, 373, 380]
RIGHT_EYE_LANDMARKS = [33, 133, 159, 145, 158, 157, 173, 246, 161, 160, 144, 163]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape  # Get frame size

            # Function to draw iris as a circle
            def draw_iris(iris_landmarks, color):
                iris_points = []
                for idx in iris_landmarks:
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    iris_points.append((x, y))

                if len(iris_points) == 5:  # Only proceed if we have all 5 points
                    center, radius = cv2.minEnclosingCircle(np.array(iris_points))
                    center = (int(center[0]), int(center[1]))
                    radius = int(radius)
                    cv2.circle(frame, center, radius, color, 2)  # Draw circle

            # Draw eye landmarks (red)
            for idx in LEFT_EYE_LANDMARKS + RIGHT_EYE_LANDMARKS:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)  # Red dots for eye landmarks

            # Draw iris as a circle (blue)
            draw_iris(LEFT_IRIS, (255, 0, 0))  # Blue for left iris
            draw_iris(RIGHT_IRIS, (255, 0, 0))  # Blue for right iris

    # Display the frame
    cv2.imshow("Iris and Eye Landmarks", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
