import cv2
import supervision as sv
import numpy as np
from ultralytics import YOLO

# Load YOLO model
# Replace the weight of desired trained model from weights folder
model = YOLO("weights\yolov9s_trained_best.pt") 

# Open webcam
cap = cv2.VideoCapture(0)


# check if the webcam is opened correctly
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

# Initialize annotators
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Run inference on webcam feed
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame, conf = 0.7)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Annotate the frame
    annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

    # Display the frame
    cv2.imshow("Webcam", annotated_frame)

    # Handle key press
    k = cv2.waitKey(1)
    if k % 256 == 27:  # ESC key
        print("Escape hit, closing...")
        break

cap.release()
cv2.destroyAllWindows()
