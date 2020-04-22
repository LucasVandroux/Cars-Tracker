import argparse
import sys

import cv2
import numpy

from utils import VehicleTracker, SimpleObjectDetector, updateVehicles

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--video_path",
    type=str,
    default="Video_traffic_tracking.mp4",
    help="Path to the video to analyse.",
)
parser.add_argument(
    "--side",
    type=str,
    default="right",
    choices=["all", "left", "right"],
    help="Select which side of the video to analyse. 'all' analyses both of the side at the same time.",
)
parser.add_argument(
    "--background_substractor_algo",
    type=str,
    default="KNN",
    choices=["KNN", "MOG2"],
    help="Select the algorithm to use for background substraction.",
)
parser.add_argument(
    "--min_contour_area",
    type=int,
    default=150,
    help="Minimum area of the bounding boxes of an object to be detected.",
)
parser.add_argument(
    "--fullscreen",
    help="Display the video in fullscreen.",
    action="store_true",
)
parser.add_argument(
    "--write_outputs",
    help="Write all the bbox in the terminal frame per frame.",
    action="store_true",
)

args = parser.parse_args()

if __name__ == "__main__":

    # Create a VideoCapture object
    capture = cv2.VideoCapture(args.video_path)

    if not capture.isOpened():
        sys.exit(f"Failed to open the video file '{args.video_path}' during the initialization.")

    # Create the object detector
    detector = SimpleObjectDetector(
        background_substractor_algo = args.background_substractor_algo, 
        min_contour_area = args.min_contour_area,
    )

    # Initialize the object detector
    detector.initializeBackground(args.video_path, side = args.side)

    list_tracked_vehicles = []

    # Set the window to fullscreen
    if args.fullscreen:
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Print instructions
    print(f"------------- VEHICLES TRACKING -------------")
    print(f" -> Press 'ESC' to quit")

    while capture.isOpened():
        ret, frame = capture.read()

        if ret:
            # Get the index of the current frame
            index_frame = capture.get(cv2.CAP_PROP_POS_FRAMES)

            middle_frame = int(frame.shape[1] / 2)

            frame_ = frame.copy()

            if args.side == "left":
                frame_[:, middle_frame:, ::] = 0
            elif args.side == "right":
                frame_[:, :middle_frame, ::] = 0
            
            # Detect the objects in the new frame
            list_detections, _ = detector.detect(frame_)

            # Update the position of the tracked vehicles
            list_tracked_vehicles, list_detections = updateVehicles(list_tracked_vehicles, list_detections)            

            # Add the remaining detection to the tracker
            for detection in list_detections:
                x, y, w, h = detection
                list_tracked_vehicles.append(VehicleTracker(x, y, w, h))

            for vehicle in list_tracked_vehicles:
                vehicle.drawOnFrame(frame)
                if args.write_outputs:
                    print(f"{index_frame}, {vehicle.id[:8]}, {vehicle.bbox.x}, {vehicle.bbox.y}, {vehicle.bbox.w}, {vehicle.bbox.h}")
            
            # Print the index_frame
            cv2.rectangle(frame, (10, 2), (200,20), (255,255,255), -1)
            cv2.putText(frame, f"{index_frame} | {args.side} | {len(list_tracked_vehicles)}", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
            
            # Display the frame
            cv2.imshow('frame', frame)

            # Wait for 'ESC' to exit the program
            keyboard = cv2.waitKey(30)
            if keyboard == 'q' or keyboard == 27:
                break

        else:
            break

    # When everything done, release the video capture object
    capture.release()

    # Destroy all the frames
    cv2.destroyAllWindows()

    # Print last line
    print(f"Vehicle tracker terminated successfully!")
    print(f"---------------------------------------------")

