import cv2
import imutils
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib

RESIZE_WIDTH = 450
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3
COUNTER=0
TOTAL=0
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(shape_predictor_path)
detector = dlib.get_frontal_face_detector()

(lStart,lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rstart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1],eye[5])
    B = dist.euclidean(eye[2],eye[4])
    C = dist.euclidean(eye[0],eye[3])
    ear = (A+B)/(2.0*C)
    return ear

def detectBlink(frame):
    global predictor
    global detector
    global lStart,lEnd,rstart,rEnd
    global RESIZE_WIDTH
    global EYE_AR_THRESH,EYE_AR_CONSEC_FRAMES,COUNTER,TOTAL
    
    frame = imutils.resize(frame,width=RESIZE_WIDTH)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    #detect face
    rects = detector(gray,0)
    for rect in rects:
        shape = predictor(gray,rect)
        shape = face_utils.shape_to_np(shape)
        
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rstart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR+rightEAR)/2.0
        
        #Draw
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame,[leftEyeHull],-1,(0,255,0),1)
        cv2.drawContours(frame,[rightEyeHull],-1,(0,255,0),1)
        
        if ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL+=1
            COUNTER=0
            
        cv2.putText(frame,"Blinks: {}".format(TOTAL),(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0,255), 2)
        cv2.putText(frame,"EAR: {:.2f}".format(ear),(300,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0,255), 2)
    
    return frame, ear, TOTAL