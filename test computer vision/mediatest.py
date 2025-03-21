import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Eye landmarks for EAR calculation
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

ear_history = deque(maxlen=10)  # Store last 10 EAR values for dynamic thresholding
eyes_closed = False  # Flag to track closure state

def calculate_ear(landmarks, eye_points):
    """Compute the Eye Aspect Ratio (EAR) given eye landmarks"""
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_points]
    
    # Compute EAR using the formula
    vertical_1 = np.linalg.norm(np.array(p2) - np.array(p6))
    vertical_2 = np.linalg.norm(np.array(p3) - np.array(p5))
    horizontal = np.linalg.norm(np.array(p1) - np.array(p4))

    return (vertical_1 + vertical_2) / (2.0 * horizontal)

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

# Matplotlib setup
plt.style.use('ggplot')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))

left_ear_values = []
right_ear_values = []
x_vals = []
frame_counter = 0  # Keep increasing x-axis values over time

def update(frame):
    global left_ear_values, right_ear_values, x_vals, frame_counter, eyes_closed

    ret, frame = cap.read()
    if not ret:
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract landmark coordinates
            landmarks = {i: (lm.x, lm.y) for i, lm in enumerate(face_landmarks.landmark)}

            # Convert to screen coordinates
            h, w, _ = frame.shape
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
                    eyes_closed = True  # Set flag to prevent repeated messages

                # Reset flag when eyes reopen
                if ear_drop < 0.1:  # If EAR stabilizes again
                    eyes_closed = False

            # Append values for plotting
            left_ear_values.append(left_ear)
            right_ear_values.append(right_ear)
            x_vals.append(frame_counter)
            frame_counter += 1  # Keep x-values increasing

            # Keep last 100 values for smooth plotting
            if len(left_ear_values) > 100:
                left_ear_values.pop(0)
                right_ear_values.pop(0)
                x_vals.pop(0)

    # Clear and replot graphs
    ax1.clear()
    ax2.clear()
    
    ax1.plot(x_vals, left_ear_values, label="Left EAR", color='blue')
    ax2.plot(x_vals, right_ear_values, label="Right EAR", color='red')
    
    ax1.set_ylim(0, 0.5)
    ax2.set_ylim(0, 0.5)
    
    ax1.legend()
    ax2.legend()
    
    ax1.set_title("Left Eye EAR")
    ax2.set_title("Right Eye EAR")

# Matplotlib animation
ani = FuncAnimation(fig, update, interval=50)

# Display the Matplotlib plot
plt.show()

# Release camera when done
cap.release()
cv2.destroyAllWindows()
