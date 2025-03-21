from imutils.video import VideoStream
import imutils
import time
import cv2
import os

sr = cv2.dnn_superres.DnnSuperResImpl_create()
path = "FSRCNN-small_x3.pb"

sr.readModel(path)
sr.setModel("fsrcnn",3)


vs = VideoStream(src=0).start()
time.sleep(0.2)

while True:
    frame = vs.read()
    
    upscaled = sr.upsample(frame)
    
    cv2.imshow("Upscaled",upscaled)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    
cv2.destroyAllWindows()
vs.stop()
