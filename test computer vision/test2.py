import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ultralytics import YOLO

# Load the YOLO model (Ensure you have a model trained to detect eyes)
model = YOLO('yolo11n.pt')  # Replace with your trained model

# Initialize webcam
cap = cv2.VideoCapture(0)

# Matplotlib figure and axis
fig, ax = plt.subplots()
x_data = np.arange(0, 100)  # Placeholder for row indices
y_data = np.zeros(100)  # Placeholder for pixel intensity values
line, = ax.plot(x_data, y_data, 'r-', lw=2)

ax.set_ylim(0, 255)
ax.set_xlim(0, 100)
ax.set_xlabel("Vertical Pixel Row")
ax.set_ylabel("Average Intensity")

def get_eye_density():
    """ Capture frame, detect eye, and compute vertical pixel density. """
    ret, frame = cap.read()
    if not ret:
        return np.zeros(100)

    # Perform object detection
    results = model(frame)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
 # Adjust based on your model
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            eye_region = frame[y1:y2, x1:x2]  # Crop the detected eye
            gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

            # Compute vertical intensity (average across width)
            vertical_intensity = np.mean(gray_eye, axis=1)  # Shape: (height,)

            # Resize to fit the graph
            resized_intensity = cv2.resize(vertical_intensity, (1, 100)).flatten()
            return resized_intensity

    return np.zeros(100)

def update(frame):
    """ Update function for the animated plot. """
    y_data[:] = get_eye_density()
    line.set_ydata(y_data)
    return line,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=True)

plt.title("Eye Vertical Pixel Intensity")
plt.show()

# Release resources after closing the plot
cap.release()
cv2.destroyAllWindows()
