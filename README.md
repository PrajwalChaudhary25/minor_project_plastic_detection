# 🌟 Plastic Detection and Classification 🌟

This project leverages YOLO and a CNN model to detect and classify different types of plastics from images and live video feeds.

## 📋 Table of Contents


- [Introduction](#Instruction)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Usage](#usage)
  - [Upload Photo for Inference](#upload-photo-for-inference)
  - [Live Video Inference](#live-video-inference)
- [Contributors](#contributors)

## 🛠️ Installation

Follow these steps to set up the project on your local machine:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/plastic-detection.git
    cd plastic-detection
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv myenv
    ```

3. **Activate the virtual environment:**

    - On Windows:
        ```bash
        myenv\Scripts\activate
        ```

    - On macOS/Linux:
        ```bash
        source myenv/bin/activate
        ```

4. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Running the Project

1. Ensure you have the necessary model files in the correct directories:
    - YOLO model weights: `yolo_weights/yolov9s_54epochs_best.pt`
    - CNN model: `CNN models/modern_CNN.keras`

2. Start the Flask application:
    ```bash
    python app.py
    ```

3. Open your web browser and go to `http://127.0.0.1:5000/`.

## 📊 Usage

### 📸 Upload Photo for Inference

1. On the homepage, click on **Choose File** to select an image file from your computer.
2. Click on **Upload** to upload the file and perform inference.
3. The results will be displayed on a new page with the detected objects and their classifications.

### 📹 Live Video Inference

1. On the homepage, click on **Start Video Inference**.
2. A new page will open with the live video feed from your webcam.
3. The detected objects and their classifications will be displayed in real-time.

## 👥 Contributors

- Jesis Upadhayaya (THA078BCT017)
- Kamal Shrestha (THA078BCT018)
- Pabin Khanal (THA078BCT027)
- Prajwal Chaudhary (THA078BCT028)

---

Feel free to reach out to us if you have any questions or suggestions! 😊
## 📧 Contact

For any questions or suggestions, feel free to reach out to us at:

- Prajwal Chaudhary: [prajwalchy25@gmail.com](mailto:prajwalchy25@gmail.com)
