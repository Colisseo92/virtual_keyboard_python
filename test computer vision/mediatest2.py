import cv2
import mediapipe as mp
import itertools

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (MediaPipe uses RGB images)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    LEFT_EYE_INDEXES = list(set(itertools.chain(*results.FACEMESH_LEFT_EYE)))
    RIGHT_EYE_INDEXES = list(set(itertools.chain(*results.FACEMESH_RIGHT_EYE)))
    # Draw face mesh if landmarks are detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)  # Draw a small green dot

    # Show the frame
    cv2.imshow('MediaPipe FaceMesh', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
