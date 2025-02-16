import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Load YOLO model (for object detection)
yolo_model = YOLO("weights/yolov9s_trained_best.pt")  # Load your YOLO model

# Load CNN model (for classification)
cnn_model = load_model("CNN models/Best.keras")  # Load your CNN model

# Class labels for plastic classification
class_labels = {0: 'HDPE', 1: 'Other', 2: 'PET', 3: 'PP', 4: 'PS'} # Adjusted based on training data

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error opening video stream")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO detection
    results = yolo_model(frame, conf=0.5)[0]  # Get first result

    for i, (x_min, y_min, x_max, y_max) in enumerate(results.boxes.xyxy):
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

        # Crop detected object
        cropped_image = frame[y_min:y_max, x_min:x_max]

        if cropped_image.size == 0:
            continue  # Skip if the cropped image is empty

        # Resize to CNN input size (e.g., 128x128)
        cropped_image = cv2.resize(cropped_image, (224, 224))

        # Normalize for CNN model
        cropped_image = cropped_image / 255.0
        cropped_image = np.expand_dims(cropped_image, axis=0)  # Add batch dimension

        # Predict with CNN
        prediction = cnn_model.predict(cropped_image)
        class_idx = np.argmax(prediction)
        class_name = class_labels[class_idx]
        confidence = prediction[0][class_idx]  # Get confidence score
        
        # Draw bounding box around detected object
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (160, 32, 240), 2)

        # Display classification label
        label = f"{class_name} ({confidence:.2f})"
        cv2.putText(frame, label, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Show the annotated frame
    cv2.imshow("Plastic Detection & Classification", frame)

    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()