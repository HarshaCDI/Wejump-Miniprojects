import os
import torch  # ✅ This line is missing in your code
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from animegan_utils import load_anime_model, preprocess_image, postprocess_tensor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load pretrained AnimeGANv2 model
model = load_anime_model('face_paint_512_v2')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    uploaded_file = request.files['image']
    filename = secure_filename(uploaded_file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'anime_{filename}')
    uploaded_file.save(input_path)

    # Convert image
    input_tensor = preprocess_image(input_path)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    result_image = postprocess_tensor(output_tensor)
    result_image.save(output_path)

    return render_template('result.html',
                           input_image=input_path,
                           output_image=output_path)

if __name__ == '__main__':
    app.run(debug=True)
