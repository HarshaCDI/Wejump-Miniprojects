from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
import librosa
import json
from tensorflow.keras.models import load_model

# Initialize Flask
app = Flask(__name__)

# Configure Upload Folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Model and Labels
model = load_model('model/genre_model.h5')
with open('model/labels.json', 'r') as f:
    genre_labels = json.load(f)

import matplotlib.pyplot as plt
import librosa.display

def extract_mel_spectrogram(audio_path, save_img_path=None):
    y, sr = librosa.load(audio_path, duration=30)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Pad or crop to 129 time frames
    if mel_db.shape[1] < 129:
        mel_db = np.pad(mel_db, ((0, 0), (0, 129 - mel_db.shape[1])), mode='constant')
    else:
        mel_db = mel_db[:, :129]

    # Normalize
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())

    # Save spectrogram image (optional)
    if save_img_path:
        plt.figure(figsize=(8, 4))
        librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel')
        plt.title('Mel Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.savefig(save_img_path)
        plt.close()

    return mel_norm.reshape(1, 128, 129, 1)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['audio']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Save path for spectrogram image
        spectro_filename = filename.rsplit('.', 1)[0] + '_spec.png'
        spectro_path = os.path.join('static/spectrograms', spectro_filename)

        # Extract features and save spectrogram
        features = extract_mel_spectrogram(filepath, save_img_path=spectro_path)

        # Predict genre
        prediction = model.predict(features)[0]
        predicted_index = int(np.argmax(prediction))
        predicted_genre = genre_labels[str(predicted_index)]

        # Fix path for web
        audio_file = filepath.replace("\\", "/")
        spectro_file = spectro_path.replace("\\", "/")

        return render_template('index.html', audio_file=audio_file, genre=predicted_genre, spectrogram=spectro_file)

    return render_template('index.html', audio_file=None, genre=None, spectrogram=None)



# Run the app
if __name__ == '__main__':
    app.run(debug=True)
