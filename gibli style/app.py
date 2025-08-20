from flask import Flask, request, render_template, send_file
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=all logs, 1=filter INFO, 2=filter WARNING, 3=filter ERROR


app = Flask(__name__)

# Load your chosen model
pipe = StableDiffusionPipeline.from_pretrained("nitrosocke/Ghibli-Diffusion", torch_dtype=torch.float16)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded = request.files["image"]
        img = Image.open(uploaded).convert("RGB").resize((512, 512))
        prompt = request.form.get("prompt", "a Ghibli-style scene")
        output = pipe(prompt=prompt, image=img if "img2img" else None, strength=0.75, guidance_scale=7.5).images[0]
        
        buf = io.BytesIO()
        output.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
