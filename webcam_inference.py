import cv2
import supervision as sv
from ultralytics import YOLO

# Load model with improved path handling
model = YOLO(r"weights\yolov9s_trained_best.pt")  # Raw string for Windows paths

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error opening video stream")
    exit()

# Initialize annotators
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # Run inference
    results = model(frame, conf=0.7, verbose=False)[0]  # verbose=False reduces logs
    detections = sv.Detections.from_ultralytics(results)
    print(detections)
    
    # Annotate
    annotated_frame = box_annotator.annotate(frame, detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections)
    
    # Display
    cv2.imshow("Webcam Inference", annotated_frame)
    
    # Exit on ESC
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()