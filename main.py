import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np

ZONE_POLYGON = np.array([
    [0, 0],
    [1280, 0],
    [1250, 720],
    [0, 720]
])

def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 Live")
    parser.add_argument(
        "--webcam-resolution", 
        default = [1280, 720],
        nargs = 2,
        type = int 
    )
    args = parser.parse_args()
    return args

def main():

    args = parse_argument()
    frame_width, frame_height = args.webcam_resolution


    cap = cv2.VideoCapture(0) #to capture video from webcam
    # cap = cv2.VideoCapture('video.mp4')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model  = YOLO("yolov8l.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    zone = sv.PolygonZone(zone = ZONE_POLYGON, frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.red)

    while True:
        ret, frame = cap.read() #cap.read is to read the data from the webcam and frame has the data and ret has the boolean value of wheather the data was shown or not
        
        result = model(frame)[0]
        detections = sv.Detections.from_yolov8(result)
        labels = [
            f"{model.names[class_id]} {confidence: 0.2f}"
            for _, confidence, class_id, _
            in detections
        ]

        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )
        
        cv2.imshow('yoloV8', frame) #imshow is use to create a window and show the data from frame which is the webcam feed
        print(ret)

        if(cv2.waitKey(30) == 27):
            break

if __name__ == "__main__":
    main() 