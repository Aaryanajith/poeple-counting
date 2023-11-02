from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np

# initiate polygon zone
polygon = np.array([
    [0, 0],
    [1919, 0],
    [1919, 1079],
    [0, 1079],
    [0, 0]
])
video_info = sv.VideoInfo.from_video_path('people_walking.mp4')
zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info.resolution_wh)

# initiate annotators
box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.white(), thickness=0, text_thickness=6, text_scale=4)

# Importing the model

model = YOLO('yolov8l.pt')

# To get video from frame

generator = sv.get_video_frames_generator('people_walking.mp4')

for frame in generator:


    # Detection

    results = model(frame, imgsz = 1280)[0]
    detections = sv.Detections.from_yolov8(results)
    detections = detections[detections.class_id == 0] # 0 is persons class
    zone.trigger(detections=detections)

    # Count the number of persons detected

    count_persons = len(detections)

    # Annotating

    box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
    labels = [f"{model.names[class_id]}{confidence:0.2f}" for _, confidence, class_id, _ in detections]
    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
    # frame = zone_annotator.annotate(scene=frame)

    # Display the annotated frame in a window
    cv2.putText(frame, f"Persons Detected: {count_persons}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Annotated Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()