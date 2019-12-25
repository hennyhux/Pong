from __future__ import print_function
import cv2 as cv
import argparse

#Python code to be loaded onto Raspberry Pi soon

def detectAndDisplay(frame):
    output = frame.copy()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #Greyscale the image
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=2, minNeighbors=5)
    for (x,y,w,h) in faces:
        cv.rectangle(output, (x,y), (x+w, y+h), (0, 0, 255), 2)
        face_roi = frame_gray[y: y+h, x: x+w] #The entire region of the face
        eye_roi_in_face = frame_gray[y: int(.7 * (y+h)), x: x+w] #The upper 70 percent of the face
        mouth_roi_in_face = frame_gray[int(.7 * (y+h)): y+h, x: x+w] #The lower 30 percent of the face
        #-- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(eye_roi_in_face, scaleFactor=1.1, minNeighbors=6) #TODO tweak this value
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.25))
            cv.circle(output, eye_center, radius, (0, 155, 0), 2)
        #-- In each face, detect smiles
        smile = smile_cascade.detectMultiScale(mouth_roi_in_face, scaleFactor=1.09, minNeighbors=260) #TODO tweak this value
        for (x3, y3, w3, h3) in smile:
            cv.rectangle(output, (x + x3, y + h - h3 - y3), (x + x3 + w3, y + h - y3), (42, 32, 54), 2)

    cv.imshow('Capture - Face detection', output)
    cv.imshow('Original', frame)


#-- 0. Load in parser and parse XML files
parser = argparse.ArgumentParser(description='Haars for Cascade Classifier.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='data/haarcascade_frontalface_default.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='data/haarcascade_eye.xml')
parser.add_argument('--smile_cascade', help='Path to smile cascade.', default='data/haarcascade_smile.xml')
parser.add_argument('--fist_cascade', help='Path to fist cascade.', default='data/haarcascade_palm.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
smile_cascade_name = args.smile_cascade
fist_cascade_name = args.fist_cascade
face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()
smile_cascade = cv.CascadeClassifier()
fist_cascade = cv.CascadeClassifier()

#-- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)
if not smile_cascade.load(cv.samples.findFile(smile_cascade_name)):
    print('--(!)Error loading smile cascade')
    exit(0)
if not fist_cascade.load(cv.samples.findFile(fist_cascade_name)):
    print('--(!)Error loading smile cascade')
    exit(0)

camera_device = args.camera
#-- 2. Read the video stream
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture source')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    if cv.waitKey(1) == 27:
        exit(0)