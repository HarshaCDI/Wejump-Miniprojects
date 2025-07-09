# from flask import Flask, render_template, request, url_for, send_from_directory
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import os

# app = Flask(__name__)
# model = MobileNetV2(weights='imagenet')
# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     uploaded_image = None
#     prediction = None

#     if request.method == 'POST':
#         if 'upload' in request.form:
#             file = request.files['file']
#             if file:
#                 filepath = os.path.join(UPLOAD_FOLDER, file.filename)
#                 file.save(filepath)
#                 uploaded_image = file.filename

#         elif 'predict' in request.form:
#             filename = request.form['filename']
#             filepath = os.path.join(UPLOAD_FOLDER, filename)

#             img = image.load_img(filepath, target_size=(224, 224))
#             img_array = image.img_to_array(img)
#             img_array = np.expand_dims(img_array, axis=0)
#             img_array = preprocess_input(img_array)

#             preds = model.predict(img_array)
#             result = decode_predictions(preds, top=1)[0][0]
#             prediction = f"{result[1]} ({round(result[2]*100, 2)}% confidence)"
#             uploaded_image = filename

#     return render_template('index.html', uploaded_image=uploaded_image, prediction=prediction)

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(UPLOAD_FOLDER, filename)

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, render_template, request, url_for, send_from_directory
# from tensorflow.keras.applications import EfficientNetB0
# from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import os

# app = Flask(__name__)

# # Load EfficientNetB0 model with ImageNet weights
# model = EfficientNetB0(weights='imagenet')

# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     uploaded_image = None
#     prediction = None

#     if request.method == 'POST':
#         if 'upload' in request.form:
#             file = request.files['file']
#             if file:
#                 filepath = os.path.join(UPLOAD_FOLDER, file.filename)
#                 file.save(filepath)
#                 uploaded_image = file.filename

#         elif 'predict' in request.form:
#             filename = request.form['filename']
#             filepath = os.path.join(UPLOAD_FOLDER, filename)

#             img = image.load_img(filepath, target_size=(224, 224))
#             img_array = image.img_to_array(img)
#             img_array = np.expand_dims(img_array, axis=0)
#             img_array = preprocess_input(img_array)

#             preds = model.predict(img_array)
#             result = decode_predictions(preds, top=1)[0][0]
#             prediction = f"{result[1]} ({round(result[2]*100, 2)}% confidence)"
#             uploaded_image = filename

#     return render_template('index.html', uploaded_image=uploaded_image, prediction=prediction)

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(UPLOAD_FOLDER, filename)

# if __name__ == '__main__':
#     app.run(debug=True)

# import os
# import cv2
# import numpy as np
# from flask import Flask, render_template, request, send_from_directory

# app = Flask(__name__)
# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# # Load EfficientNet model
# model = cv2.dnn.readNetFromONNX('efficientnet_b0.onnx')

# # Load ImageNet labels
# with open('imagenet_labels.txt') as f:
#     labels = f.read().splitlines()

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     uploaded_image = None
#     prediction = None

#     if request.method == 'POST':
#         file = request.files['file']
#         if file:
#             filename = file.filename
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(filepath)
#             uploaded_image = filename

#             # Load and preprocess image
#             img = cv2.imread(filepath)
#             blob = cv2.dnn.blobFromImage(img, scalefactor=1.0/255, size=(224, 224),
#                                          mean=(0, 0, 0), swapRB=True, crop=False)

#             # Run inference
#             model.setInput(blob)
#             output = model.forward()
#             class_id = int(np.argmax(output))
#             confidence = float(output[0][class_id])

#             prediction = f"{labels[class_id]} ({round(confidence*9, 2)}% confidence)"

#     return render_template('index.html', uploaded_image=uploaded_image, prediction=prediction)

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# if __name__ == '__main__':
#     app.run(debug=True)

import os
import cv2
import numpy as np
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Create folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load EfficientNet model for image classification
image_model = cv2.dnn.readNetFromONNX('efficientnet_b0.onnx')
with open('imagenet_labels.txt') as f:
    image_labels = f.read().splitlines()

# Load MobileNet SSD model for video object detection
video_model = cv2.dnn.readNetFromCaffe(
    'models/MobileNetSSD_deploy.prototxt',
    'models/MobileNetSSD_deploy.caffemodel'
)
video_labels = ["background", "aeroplane", "bicycle", "bird", "boat",
                "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                "dog", "horse", "motorbike", "person", "pottedplant",
                "sheep", "sofa", "train", "tvmonitor"]

@app.route('/', methods=['GET', 'POST'])
def index():
    uploaded_image = None
    prediction = None

    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file:
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            uploaded_image = filename

            # Read and preprocess image
            img = cv2.imread(filepath)
            if img is None:
                prediction = "Could not read image!"
            else:
                blob = cv2.dnn.blobFromImage(img, scalefactor=1.0/255, size=(224, 224),
                                             mean=(0, 0, 0), swapRB=True, crop=False)
                image_model.setInput(blob)
                output = image_model.forward()
                class_id = int(np.argmax(output))
                confidence = float(output[0][class_id])
                prediction = f"{image_labels[class_id]} ({round(confidence * 100, 2)}% confidence)"

    return render_template('index.html', uploaded_image=uploaded_image, prediction=prediction, video_path=None)

@app.route('/video', methods=['GET', 'POST'])
def video_page():
    video_path = None

    if request.method == 'POST' and 'video' in request.files:
        file = request.files['video']
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        cap = cv2.VideoCapture(filepath)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        result_path = os.path.join(RESULT_FOLDER, 'result.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            video_model.setInput(blob)
            detections = video_model.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.4:
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    (startX, startY, endX, endY) = box.astype("int")
                    label = f"{video_labels[idx]}: {round(confidence * 100, 2)}%"
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(frame, label, (startX, startY - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            out.write(frame)

        cap.release()
        out.release()
        video_path = 'result.mp4'

    return render_template('video.html', video_path=video_path)

@app.route('/multiobject', methods=['GET', 'POST'])
def multiobject():
    detected_image = None

    if request.method == 'POST' and 'image' in request.files:
        file = request.files['image']
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        if img is None:
            return render_template('multiobject.html', error="Could not read image.")

        height, width = img.shape[:2]

        # Prepare image for object detection
        blob = cv2.dnn.blobFromImage(img, 0.007843, (300, 300), 127.5)
        video_model.setInput(blob)
        detections = video_model.forward()

        # Draw detections with enhanced labels
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.4:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                label = f"{video_labels[idx]}: {round(confidence * 100, 2)}%"

                # Get text size for background rectangle
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (startX, startY - text_height - 10),
                              (startX + text_width + 4, startY), (0, 0, 255), -1)

                # Draw bounding box and label
                cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(img, label, (startX + 2, startY - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Save the result image
        result_path = os.path.join(RESULT_FOLDER, 'multiobject_result.jpg')
        cv2.imwrite(result_path, img)
        detected_image = 'multiobject_result.jpg'

    return render_template('multiobject.html', detected_image=detected_image)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)




@app.route('/static/<path:filename>')
def serve_video(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
