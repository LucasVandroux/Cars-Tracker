# Simple Vehicle Tracking Without Deep Learning

<details><summary>Table of Contents</summary><p>

1. [Installation](https://github.com/LucasVandroux/Interview-Huawei-London#installation)
2. [Usage](https://github.com/LucasVandroux/Interview-Huawei-London#usage)
3. [Vehicle Tracker](https://github.com/LucasVandroux/Interview-Huawei-London#vehicle-tracker)
4. [Outlook](https://github.com/LucasVandroux/Interview-Huawei-London#outlook)

</p></details><p></p>

In this repository, we are proposing a simple vehicle tracker for static cameras that requires minimum computation power and no prior training of the algorithms used as it relies solely on traditional computer vision methods.

![Demo of the Vehicle Tracking](images/vehicles_tracker_demo.gif "Demo of the Vehicle Tracking")

_Demo of the tracker, tracking vehicles on the right track._

## Installation

We recommend using an isolated Python environment, such as [venv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io/en/latest/) with at least **Python 3.6**. Then, use the following lines of code:

```
git clone https://github.com/LucasVandroux/Interview-Huawei-London.git
cd Interview-Huawei-London
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

## Usage

Once the installation process is finished, use the `main.py` script to track the vehicles on the video of your choice.

To get more information about the `main.py` script, use the following line in your terminal:

```
python main.py --help
```

To have a first impression on the performances of the tracker, use the following line:

```
python main.py --fullscreen --side all
```

To get the analysis on one specific side of the road and have all the detections and tracking information frame per frame written in the terminal, use the following line:

```
python main.py --write_outputs --side right
```

### Outputs

If the flag `--write_outputs` is used, the trackers' information will be written frame by frame in the terminal. It uses the [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) format with the following columns:
1. `index_frame`: Index of the frame the detection has been made on.
2. `tracker_id`: Unique identifier for the tracker of a specific vehicle.
3. `bbox_x`: x coordinate of the top-left corner of the bounding box containing the vehicle
4. `bbox_y`: y coordinate of the top-left corner of the bounding box containing the vehicle
5. `bbox_w`: width in pixels of the bounding box containing the vehicle. 
6. `bbox_h`: height in pixels of the bounding box containing the vehicle.

#### Example:
```
index_frame, tracker_id, bbox_x, bbox_y, bbox_w, bbox_h
1, f74adaf2, 552, 276, 23, 17
1, f74ade4e, 515, 242, 56, 48
1, f74ae092, 430, 215, 35, 35
1, f74ae286, 367, 199, 31, 30
1, f74ae452, 337, 150, 16, 15
1, f74ae600, 373, 133, 35, 32
2, f74adaf2, 552, 276, 23, 17
2, f74ade4e, 515, 242, 56, 48
2, f74ae092, 430, 215, 35, 35
2, f74ae286, 367, 199, 31, 30
2, f74ae452, 337, 150, 16, 15
2, f74ae600, 373, 133, 35, 32
...
```

## Vehicle Tracker
The vehicle tracker implemented in this repository can be divided into two main components, an object detector, and a tracking method.

### Object Detector
The object detector uses a background subtractor algorithm to detect the moving objects in the video and return the bounding boxes around them. The detector leverages different algorithms already available in OpenCV, including the [KNN background subtractor](https://docs.opencv.org/3.4/db/d88/classcv_1_1BackgroundSubtractorKNN.html) that is the default method for this tracker.

The object detector was inspired by the method described in [Real-time Moving Vehicle Detection, Tracking, and
Counting System Implemented with OpenCV](https://ieeexplore.ieee.org/document/6920557/authors) by Da Li, Bodong Liang, and Weigang Zhang.

### Tracking
The tracking algorithm is matching the bounding boxes in the new frame with the one from the previous frame if their [Intersection over Union](https://en.wikipedia.org/wiki/Jaccard_index) is higher than a certain threshold. In case a bounding box from the previous frame can't be matched with one in the new frame, then the tracker on this specific object is considered lost and therefore removed.

### Limitations
This approach has some significant limitations. First, it can only detect moving objects and, therefore, will have trouble monitoring a massive traffic jam, for example. Secondly, it does not only recognize moving vehicles in the video but any moving object/pattern. Therefore, some shadows cast by large trucks or big moving trees are considered as vehicles too. Finally, it can't handle the occlusion between two moving objects as those two objects will be considered as one single object.

## Outlook
The limitations listed above could significantly be mitigated with the use of a deep learning object detector specially trained on recognizing vehicles. Architecture such as [YOLO](https://pjreddie.com/darknet/yolo/) could be a good fit for this task. To improve the tracking itself, without using Deep Learning or any other external data sources, one could also consider using one of the trackers in the [OpenCV API](https://docs.opencv.org/3.4/d9/df8/group__tracking.html). The object detector could be used to select the ROIs to track automatically. However, using too many of those trackers might decrease the efficiency of the overall system as they are relatively computationally expensive.



