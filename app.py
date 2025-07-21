from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response, flash
import os
import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load models
yolo_model = YOLO("yolo_weights\Yoloppps.pt")
cnn_model = tf.keras.models.load_model("CNN models/Pabin_Model.keras")
# class_labels = {0: 'HDPE', 1: 'Other', 2: 'PET', 3: 'PP', 4: 'PS'}
class_labels = {0: 'HDPE', 1: 'PET', 2: 'PP', 3: 'PS'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return redirect(url_for('inference', filename=filename))
    flash('Invalid file type')
    return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/inference/<filename>')
def inference(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    original_img = cv2.imread(filepath)
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # YOLO inference
    yolo_result = yolo_model(source=filepath, conf=0.5)
    cropped_images = []
    labels = []

    # Iterate over detected objects
    for detection in yolo_result[0].boxes:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        cropped_img = original_img_rgb[y1:y2, x1:x2]

        # Preprocess the cropped image for CNN
        img = cv2.resize(cropped_img, (224, 224))
        img_array = img / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # CNN inference
        cnn_prediction = cnn_model.predict(img_array)
        predicted_class_index = np.argmax(cnn_prediction)
        predicted_label = class_labels[predicted_class_index]

        # Store cropped image and label
        cropped_images.append(cropped_img)
        labels.append(predicted_label)

    # Display cropped images in a grid with labels
    num_columns = 4
    num_rows = max(1, (len(cropped_images) + num_columns - 1) // num_columns)
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 5 * num_rows))
    plt.subplots_adjust(hspace=0.5)  # Add vertical space between rows

    for i, (img, label) in enumerate(zip(cropped_images, labels)):
        row = i // num_columns
        col = i % num_columns
        ax = axes[row, col] if num_rows > 1 else axes[col]
        ax.imshow(img)
        ax.set_title(label)
        ax.axis('off')

    # Hide any unused subplots
    for j in range(len(cropped_images), num_rows * num_columns):
        row = j // num_columns
        col = j % num_columns
        ax = axes[row, col] if num_rows > 1 else axes[col]
        ax.axis('off')

    output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'output.png')
    plt.savefig(output_filepath)
    plt.close()

    return render_template('inference.html', filename='output.png')

@app.route('/video_inference')
def video_inference():
    return render_template('video_inference.html')

def generate_video_frames():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Error opening video stream"

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # # Calculate FPS
        # current_time = time.time()
        # fps = 1 / (current_time - prev_time)
        # prev_time = current_time

        # Run YOLO detection
        results = yolo_model(frame, conf=0.5)[0]  # Get first result

        for i, (x_min, y_min, x_max, y_max) in enumerate(results.boxes.xyxy):
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

            # Crop detected object
            cropped_image = frame[y_min:y_max, x_min:x_max]

            if cropped_image.size == 0:
                continue  # Skip if the cropped image is empty

            # Resize to CNN input size (e.g., 224x224)
            cropped_image = cv2.resize(cropped_image, (224, 224))

            # Normalize for CNN model
            cropped_image = cropped_image / 255.0
            cropped_image = np.expand_dims(cropped_image, axis=0)  # Add batch dimension

            # Predict with CNN
            prediction = cnn_model.predict(cropped_image)
            class_idx = np.argmax(prediction)
            confidence = prediction[0][class_idx]  # Get confidence score

            # Apply confidence threshold
            if confidence >= 0.40:
                class_name = class_labels[class_idx]
            else:
                class_name = "Unknown"

            # Draw bounding box around detected object
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (160, 32, 240), 2)

            # Display classification label
            label = f"{class_name} ({confidence:.2f})"
            cv2.putText(frame, label, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # yield (b'--frame\r\n'
        #        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_video_inference', methods=['POST'])
def start_video_inference():
    return redirect(url_for('video_inference'))

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
