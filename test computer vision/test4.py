from l2cs import Pipeline, render
import cv2
import torch

gaze_pipeline = Pipeline(
    weights='L2CSNet_gaze360.pkl',
    arch='ResNet50',
    device=torch.device('cpu') # or 'gpu'
)
 
cap = cv2.VideoCapture()
_, frame = cap.read()    

# Process frame and visualize
results = gaze_pipeline.step(frame)
frame = render(frame, results)