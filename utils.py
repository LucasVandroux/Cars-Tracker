from collections import namedtuple
import uuid

import cv2
import numpy as np
from tqdm import tqdm


class SimpleObjectDetector:
    """ Simple object detector for a static camera

    Simple object detector returning a list of bounding boxes of objects 
    that have been detected using a background substraction algorithm.
    """

    def __init__(
        self, background_substractor_algo: str = "KNN", min_contour_area: int = 250,
    ):
        """ Constructor for the SimpleObjectDetector
        
        Args:
            background_substractor_algo (str: "KNN"): background substraction algorithm to use      
                (MOG2 or KNN). More info here: 
                https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html
            min_contour_area (int: 250): minimum area of the bounding boxes of an object to be 
                detected.
        """
        self.min_contour_area = min_contour_area

        # Set the background substractor algoritm
        if background_substractor_algo == "MOG2":
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
                detectShadows=True
            )
        else:
            self.background_subtractor = cv2.createBackgroundSubtractorKNN(
                detectShadows=True
            )

    def initializeBackground(
        self, video_path: str, num_frames_for_initialization: int = 200, side: str = "all"
    ):
        """ Initialize the background image used by the background substracor algorithm

        It is not required to use it but it can improve the detection performances in the first 
        frames of a new video.

        Args:
            video_path (str): path to the video the background substractor algorithm should be 
                initialized on.
            num_frames_for_initialization (int: 200): number of frames to use for the 
            initialization.
            side (str: "all"): side of the image that will be analyzed

        Returns:
            (bool): True if the initialization was successful or False otherwise.

        """
        # Create a VideoCapture object
        capture = cv2.VideoCapture(video_path)

        # Check if the VideoCapture was created correctly.
        if not capture.isOpened():
            print(
                f"Failed to open the video file '{video_path}' during the initialization of the SimpleObjectDetector."
            )
            return False

        print(f"Initialization of the SimpleObjectDetector...")
        # Initialize the progress bar
        pbar = tqdm(total=num_frames_for_initialization)

        # Start the loop to initialize the background
        while capture.get(cv2.CAP_PROP_POS_FRAMES) < num_frames_for_initialization:
            ret, frame = capture.read()

            # If the frame was captured correctly
            if ret:
                # Find the middle of the frame
                middle_frame = int(frame.shape[1] / 2)
                
                # Black out the side that is not observed
                if side == "left":
                    frame[:, middle_frame:, ::] = 0
                elif side == "right":
                    frame[:, :middle_frame, ::] = 0

                # Blur the frame the same way as the detect function
                frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)

                # Update the background
                _ = self.background_subtractor.apply(frame_blurred)

                # Update the progress bar
                pbar.update(capture.get(cv2.CAP_PROP_POS_FRAMES))

            # In case the video was too short or another problem happened
            else:
                # Terminate the progress bar
                pbar.close()
                print(
                    f"Failed to finished the initialization of the SimpleObjectDetector (Exited on frame number: {cv2.CAP_PROP_POS_FRAMES})."
                )
                return False

        # Terminate the progress bar
        pbar.close()
        print(f"Successfully initialized the SimpleObjectDetector.")
        return True

    def detect(self, frame):
        """ Perform object detection on a specific frame

        Args:
            frame(np.array): frame from OpenCV where the detection needs to be performed.

        Returns:
            list_detections (list[(x: int, y:int, w:int, h:int)]): list of bounding boxes of the detected objects. A bounding boxe is a tuple (x, y, w, h) where (x,y) are the coordinates of the top-left corner of the bounding box and w its width and h its height in pixels.
            foreground_mask_ (np.array): binary mask used to detect the objects.
        """
        # --- PRE-PROCESSING ---
        # Blur the frame
        frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # Use the background substractor to extract the foreground
        foreground_mask = self.background_subtractor.apply(frame_blurred)

        # Remove the shadows from the foreground
        foreground_mask[foreground_mask < 255] = 0

        # Edges detection
        edges = cv2.Canny(foreground_mask, 100, 200)

        # Open operation to remove the small detection
        foreground_mask_ = cv2.erode(foreground_mask, None, iterations=1)
        foreground_mask_ = cv2.dilate(foreground_mask_, None, iterations=1)

        # Combine the edges detection and the foreground_mask_
        foreground_mask_ = foreground_mask_ + edges
        foreground_mask_ = cv2.dilate(foreground_mask_, None, iterations=1)

        # Find the closed contours and fill them
        cnts = cv2.findContours(
            foreground_mask_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            cv2.drawContours(foreground_mask_, [c], 0, (255, 255, 255), -1)

        # --- DETECTION ---
        contours, hierarchy = cv2.findContours(
            foreground_mask_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Convert each found contour into a bounding box
        list_detections = []
        for contour in contours:
            if cv2.contourArea(contour) >= self.min_contour_area:
                # Extract the bounding box around the detection
                bbox = cv2.boundingRect(contour)
                list_detections.append(bbox)

        # list of bounding boxes of objects detected
        return list_detections, foreground_mask_


def updateVehicles(list_vehicles, list_detections):
    """ Update the vehicles with a new bounding box if it matches a new detection

    Args:
        list_vehicles (list[VehicleTracker]): list of VehicleTracker objects
        list_detections (list[(x: int, y:int, w:int, h:int)]): list of bounding boxes. A bounding boxe is a tuple (x, y, w, h) where (x,y) are the coordinates of the top-left corner of the bounding box and w its width and h its height in pixels.

    Returns:
        list_vehicles (list[VehicleTracker]): list of VehicleTracker that could be match with a new boudning box. Return the input list_vehicles if no matching is needed.
        list_detections (list[(x: int, y:int, w:int, h:int)]): list of bounding boxes that haven't been matched to a vehicle.
    """
    # Initialize the array to store of the Intersection Over Union (IoU) between the bounding boxes
    # of the vehicles and the new detections.
    iou_vehicles_detections = np.zeros([len(list_vehicles), len(list_detections)])

    # Fill up the array with all the IoU values (row -> vehicles, column -> detections)
    for idx, vehicle in enumerate(list_vehicles):
        for jdx, detection in enumerate(list_detections):
            iou_vehicles_detections[idx, jdx] = vehicle.computeIouWith(detection)

    # If the numpy array is not empty
    if iou_vehicles_detections.size:
        # Initialize the list to keep track of detections that have been matched to a vehicle
        list_index_detections_tracked = []
        # Initialize the list to keep track of the vehicle that have been updated
        list_vehicles_to_keep = []

        # Loop over the array until every vehicle has been matched
        while np.sum(iou_vehicles_detections):
            # Find coordinates of the max value in the array
            idx_vehicle, idx_detection = np.unravel_index(
                np.argmax(iou_vehicles_detections), iou_vehicles_detections.shape
            )

            # Update the vehicle bbox
            bbox_x, bbox_y, bbox_w, bbox_h = list_detections[idx_detection]
            list_vehicles[idx_vehicle].updateBbox(bbox_x, bbox_y, bbox_w, bbox_h)
            list_vehicles_to_keep.append(list_vehicles[idx_vehicle])

            # Update the list with the detection that have been attributed to a vehicle
            list_index_detections_tracked.append(idx_detection)

            # Set to zero the row of the vehicle and the column of the detection
            iou_vehicles_detections[idx_vehicle, :] = 0
            iou_vehicles_detections[:, idx_detection] = 0

        # Remove the detections that have been matched to a vehicle already
        list_detections = [
            detection
            for idx, detection in enumerate(list_detections)
            if idx not in list_index_detections_tracked
        ]
        list_vehicles = list_vehicles_to_keep

    return list_vehicles, list_detections


BoundingBox = namedtuple("BoundingBox", "x y w h")
# where (x,y) are the coordinates of the top-left corner of the bounding box and w its width and h its height in pixels.


class VehicleTracker:
    """ Vehicle Tracker

    Class responsible to track each vehicle individually.

    """

    def __init__(self, x: int, y: int, w: int, h: int):
        """ Initialize the Vehicle Tracker

        Args:
            x (int): x coordinate of the top-left corner of the inital bbox containing the vehicle
            y (int): y coordinate of the top-left corner of the inital bbox containing the vehicle
            w (int): width of the inital bbox containing the vehicle
            h (int): height of the inital bbox containing the vehicle

        """
        # Initialize the list of centroids of the bboxes
        self.list_centroids = []
        # Update the current bbox and the list of centroids
        self.updateBbox(x, y, w, h)
        # Define a unique id for this tracker
        self.id = str(uuid.uuid1())

    def computeIouWith(self, bbox, min_iou=0.5):
        """ Compute the IoU

        Compute the Intersection Over Union (IoU) between the current bounding box containing the vehicle and another bounding box.

        Args:
            bbox ((x: int, y:int, w:int, h:int)): Bounding to calculate the IoU with the current bbox of the vehicle. A bounding boxe is a tuple (x, y, w, h) where (x,y) are the coordinates of the top-left corner of the bounding box and w its width and h its height in pixels.
            min_iou (int: 0.5): minimum under which a the value of the IoU will be set to 0

        Returns:
            iou (float): value of the Intersection over Union between the current bbox of the vehicle and the bbox given as an input.

        """
        bbox_x, bbox_y, bbox_w, bbox_h = bbox
        bbox_area = bbox_w * bbox_h

        # find coordinates of the intersection
        xx1 = np.maximum(self.bbox.x, bbox_x)
        yy1 = np.maximum(self.bbox.y, bbox_y)
        xx2 = np.minimum(self.bbox.x + self.bbox.w, bbox_x + bbox_w)
        yy2 = np.minimum(self.bbox.y + self.bbox.h, bbox_y + bbox_h)

        # compute are of the intersection
        intersection = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)

        # compute the IoU
        iou = intersection / (bbox_area + (self.bbox.w * self.bbox.h) - intersection)

        # set the iou to zero if it is inferior to a certain threshold.
        if iou < min_iou:
            iou = 0

        return iou

    def updateBbox(self, x: int, y: int, w: int, h: int):
        """ Update the current Bounding Box of the tracker and the list of centroids

        Args:
            x (int): x coordinate of the top-left corner of the bbox
            y (int): y coordinate of the top-left corner of the bbox
            w (int): width of the bbox
            h (int): height of the inital bbox

        """
        self.bbox = BoundingBox(x, y, w, h)
        self.list_centroids.append(self.computeCentroid(self.bbox))

    def computeCentroid(self, bbox):
        """ Compute the center of a bounding box

        Args:
            bbox (BoundingBox): bounding box to compute the center from

        Returns:
            (centroid_x, centroid_y) ((int, int)): tuple containing the coordinates of the centroid
        """
        centroid_x = int(bbox.x + (bbox.w / 2))
        centroid_y = int(bbox.y + (bbox.h / 2))
        return (centroid_x, centroid_y)

    def drawOnFrame(self, frame, color=(0, 255, 0)):
        """ Draw the bbox and the previous centroid on an image

        Args:
            frame (np.array): image to draw the bbox and the previous centroid on.
            color ((B:int, G:int, R:int)): BGR values for the color to draw with.
        """
        # Draw the centroids
        for idx, centroid in enumerate(self.list_centroids):
            cv2.circle(frame, centroid, 2, color, -1)

        # Draw the bounding box
        cv2.rectangle(
            frame,
            (self.bbox.x, self.bbox.y),
            (self.bbox.x + self.bbox.w, self.bbox.y + self.bbox.h),
            color,
            2,
        )
