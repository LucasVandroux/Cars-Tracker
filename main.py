import argparse
import sys

import cv2
import numpy

from utils import VehicleTracker, SimpleObjectDetector

# Color use to draw and write
color = (0, 255, 0)

min_contourArea = 250

# import video
video_path = 'Video_traffic_tracking.mp4'

# Create a VideoCapture object
capture = cv2.VideoCapture(video_path)

if not capture.isOpened():
    sys.exit(f"Failed to open the video file '{video_path}' during the initialization.")

# Create the simple object detector
detector = SimpleObjectDetector(
    backgroundSubstractorAlgo = 'KNN', 
    min_contourArea = 250,
)

list_tracked_vehicle = []

frame_index = 0

while capture.isOpened():
    ret, frame = capture.read()

    if ret:
        # Detect the objects in the new frame
        list_detections = detector.detect(frame)

        # Draw the bounding boxes of the detected objects on the frame
        for detection in list_detections:
            x, y, w, h = detection
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # TODO Smaller contour can be added to the closest bigger detection
         
        cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        
        cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        
        cv2.imshow('Frame', frame)
        # cv2.imshow('FG Mask', fgMask)
        # cv2.imshow('blob', blob)
        # cv2.imshow('edges', edges)
        # cv2.imshow('out', out)
    
        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break

    else:
        break

    frame_index += 1

# When everything done, release the video capture object
capture.release()

# Closes all the frames
cv2.destroyAllWindows()
