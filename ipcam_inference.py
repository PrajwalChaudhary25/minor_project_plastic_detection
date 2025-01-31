import cv2
import supervision as sv
import numpy as np
from ultralytics import YOLO

# Replace with your IP camera feed URL
url = "http://192.168.254.226:8080/video"  # put your IP camera feed URL here


# Initialize annotators
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Load YOLO model
model = YOLO("yolov9t_trained_best.pt")

# Open IP camera video feed
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

# Initialize annotators
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Annotate the frame
    annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

    # Display the frame
    cv2.imshow("IP Camera Feed", annotated_frame)

    # Handle key press
    k = cv2.waitKey(1)
    if k % 256 == 27:  # ESC key
        print("Escape hit, closing...")
        break

cap.release()
cv2.destroyAllWindows()
