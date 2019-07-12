import cv2
import sys
from darkflow.net.build import TFNet
import numpy as np
import math


minor_ver = 4
bboxes = []

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD',
                 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']


def getTracker(tracker_type):
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
    return tracker


def getVideo():
    video = cv2.VideoCapture("vids/DJI_0083.MP4")

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    return video


def getFrame(video):
    # Read first frame
    ok, frame = video.read()
    frame = cv2.resize(frame, (1600, 900))
    if not ok:
        print('Cannot read video file')
        sys.exit()
    return frame


def getPredictions(results, confidence=0.5):
    print("confidence", confidence)
    predictions = []
    for result in results:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']
        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        result_confidence = result['confidence']
        if result_confidence > confidence:
            width = abs(top_x - btm_x)
            height = abs(top_y - btm_y)
            predictions.append((top_x, top_y, width, height))

    return predictions


if __name__ == '__main__':
    # tracker = getTracker(tracker_types[2])
    tracker = cv2.MultiTracker_create()
    video = getVideo()

    frame = getFrame(video)

    # collect data
    '''
    for i in range(3):
        bbox = cv2.selectROI(frame, False)
        tracker.add(getTracker(tracker_types[2]), frame, bbox)
    '''
    options = {"model": "cfg/yolo_ls.cfg",
               "batch": 8,
               "epoch": 1000,
               "gpu": 1.0,
               "train": True,
               "annotation": "./annotations/",
               "dataset": "./images/",
               "load": -1
               }

    tfnet = TFNet(options)
    results = tfnet.return_predict(frame)
    predictions = getPredictions(results)
    for bbox in predictions:
        print(bbox)
        tracker.add(getTracker(tracker_types[2]), frame, bbox)

    # play video
    while True:
        frame = getFrame(video)
        ok, bboxes = tracker.update(frame)

        for bbox in bboxes:
            print(bbox)
            # Draw bounding box
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            else:
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                results = tfnet.return_predict(frame)
                predictions = getPredictions(results)
                for bbox in predictions:
                    print(bbox)
                    tracker.add(getTracker(tracker_types[2]), frame, bbox)

        # Display tracker type on frame
        cv2.putText(frame, tracker_types[2] + " Tracker", (100, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
