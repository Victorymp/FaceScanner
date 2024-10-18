from __future__ import print_function
import cv2 as cv
import argparse
from pynput.mouse import Controller #pip install pynput
def mouse_box(event, x, y, flags, param):
    mouse = Controller()
    if event == cv.EVENT_LBUTTONDOWN:
        print(mouse.position[0])
        cv.imwrite("frame.jpg", param)

def mouse_hover(x):
    position = Controller().position
    print(x)
    if position[0] > x:
        print('inline')


def detectAndDisplay(frame):
    # Converts the frame colour to grayscale for easier detection
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
    # Equalizes the image using a hist
    frame_gray = cv.equalizeHist(frame_gray)
    # Mouse used as a controller
    mouse = Controller()
    #-- Detect faces in the frame 
    faces = face_cascade.detectMultiScale(frame_gray)
    # Draws a box around the face
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        color = (0, 0, 0)
        start = (x,y)
        end = (x+w,y+h)
        current = mouse.position
        # print(x)
        # cv.setMouseCallback('Capture - Face detection',mouse_hover(x,y,w,h),param=frame)
        #if mouse
        print(current , " -- mouse position ")
        print(start)
        frame = cv.rectangle(frame, start, end, color,1)
        faceROI = frame_gray[y:y+h,x:x+w]
        #-- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
    cv.imshow('Capture - Face detection', frame)
    
    
parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='data/haarcascades/haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()
#-- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)
camera_device = args.camera
#-- 2. Read the video stream
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    # cv.setMouseCallback('Capture - Face detection',mouse_box,param=frame)
    if cv.waitKey(10) == 27:
        break