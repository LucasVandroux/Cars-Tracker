from collections import namedtuple
import cv2
import numpy as np

# class BoundingBox():
#     def __init__(self, xmin:int, ymin:int, width:int, height:int):
#         self.xmin = xmin
#         self.ymin = ymin
#         self.width = width
#         self.height = height
#         self.xmax = self.xmin + self.width
#         self.ymax = self.ymin + self.height

BoundingBox = namedtuple('BoundingBox', 'x y w h')
    
class VehicleTracker():
    def __init__(self, x:int, y:int, w:int, h:int):
        self.current_bbox = BoundingBox(x, y, w , h)

        # Initialize the tracker
        # self.tracker = cv2.TrackerCSRT_create()
        # self.tracker_success = self.tracker.init(frame, (x, y, w, h))
    
    def update(self, x:int, y:int, w:int, h:int):
        self.current_bbox = BoundingBox(x, y, w , h)

class SimpleObjectDetector():
    def __init__(self, backgroundSubstractorAlgo:str = 'KNN', min_contourArea:int = 250):
        
        self.min_contourArea = min_contourArea

        if backgroundSubstractorAlgo == 'MOG2':
            self.backgroundSubtractor = cv2.createBackgroundSubtractorMOG2(detectShadows = True)
        else:
            self.backgroundSubtractor = cv2.createBackgroundSubtractorKNN(detectShadows = True)

        #TODO run the background substraction for 50 frames before starting to make sure we have the best background reference possible

    def detect(self, frame):
        # --- PRE-PROCESSING ---
        # Blur the frame
        frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # Use the background substractor to extract the foreground
        foregroundMask = self.backgroundSubtractor.apply(frame_blurred)
        
        # Remove the shadows from the foreground
        foregroundMask[foregroundMask < 255] = 0

        # Edges detection
        edges = cv2.Canny(foregroundMask,100,200)

        # Open operation to remove the small detection
        foregroundMask_ = cv2.erode(foregroundMask, None, iterations=1)
        foregroundMask_ = cv2.dilate(foregroundMask_, None, iterations=1)

        # Combine the edges detection and the foregroundMask_
        foregroundMask_ = foregroundMask_ + edges

        # Find the closed contours and fill them
        cnts = cv2.findContours(foregroundMask_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            cv2.drawContours(foregroundMask_,[c], 0, (255,255,255), -1)

        # --- DETECTION ---
        contours, hierarchy = cv2.findContours(foregroundMask_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Convert each found contour into a bounding box
        list_detections = []
        for contour in contours:
            if cv2.contourArea(contour) >= self.min_contourArea:
                # Extract the bounding box around the detection
                bbox = cv2.boundingRect(contour)
                list_detections.append(bbox)
        
        # list of bounding boxes of objects detected
        return list_detections