import argparse
import sys

import cv2
import numpy

# Color use to draw and write
color = (0, 255, 0)

min_contourArea = 250

# import video
video_path = 'Video_traffic_tracking.mp4'

# Create a VideoCapture object
capture = cv2.VideoCapture(video_path)

if not capture.isOpened():
    sys.exit(f"Failed to open the video file '{video_path}' during the initialization.")

# backSub = cv2.createBackgroundSubtractorMOG2()
backSub = cv2.createBackgroundSubtractorKNN(detectShadows = True)

while capture.isOpened():
    ret, frame = capture.read()

    if ret:
        # TODO maybe blur the frame before doing background substraction
        frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        #TODO run the background substraction for 50 frames before starting to make sure we have the best background reference possible
        fgMask = backSub.apply(frame_blurred)
        # Remove the shadows
        fgMask[fgMask < 200] = 0

        # edge detection
        edges = cv2.Canny(fgMask,100,200)
        # edges = cv2.erode(out, None, iterations=1) + cv2.dilate(edges, None, iterations=1)
        # edges = cv2.dilate(edges, None, iterations=2)

        # Remove small detections and fill the holes in the bigger ones
        # out = cv2.dilate(fgMask, None, iterations=1)
        # out = cv2.erode(out, None, iterations=2)
        # out = cv2.dilate(out, None, iterations=2)

        # Open operation
        blob = cv2.erode(fgMask, None, iterations=1)
        blob = cv2.dilate(blob, None, iterations=1)

        # combine the edges detection and the blobs
        out = blob + edges
        cnts = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            cv2.drawContours(out,[c], 0, (255,255,255), -1)

        # Blob detection
        contours, hierarchy = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            for contour in contours:
                if cv2.contourArea(contour) >= min_contourArea:
                    # Extract the bounding box around the detection
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # TODO Smaller contour can be added to the closest bigger detection
         
        cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        
        cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        
        cv2.imshow('Frame', frame)
        # cv2.imshow('FG Mask', fgMask)
        # cv2.imshow('blob', blob)
        # cv2.imshow('edges', edges)
        cv2.imshow('out', out)
    
        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break

    else:
        break

# When everything done, release the video capture object
capture.release()

# Closes all the frames
cv2.destroyAllWindows()
