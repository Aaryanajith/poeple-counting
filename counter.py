from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np

# initiate polygon zone
polygon = np.array([
    [1900, 1250],
    [2350, 1250],
    [3500, 2160],
    [1250, 2160]
])
video_info = sv.VideoInfo.from_video_path('people_walking.mp4')
zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info.resolution_wh)

# Importing the model

model = YOLO('yolov8l.pt')

# To get video from frame

generator = sv.get_video_frames_generator('people_walking.mp4')
iterator = iter(generator)
frame = next(iterator)

# Detection

results = model(frame, imgsz = 1280)[0]
detections = sv.Detections.from_yolov8(results)
detections = detections[detections.class_id == 0]

# Annotating

box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
labels = [f"{model.names[class_id]}{confidence:0.2f}" for _, confidence, class_id, _ in detections]
frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

# Display the annotated frame in a window
cv2.imshow('Annotated Frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()