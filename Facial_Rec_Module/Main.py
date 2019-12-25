import cv2 as cv
import numpy as np
import time

#--Step 0: Import YOLO
net = cv.dnn.readNet(model='data/yolov3-tiny.weights', )
classes = []
with open('data/coco.names.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))

