from flask import Flask, render_template, request
from PIL import Image
import torch
from torchvision import transforms
import sys
from torchvision.transforms.functional import InterpolationMode
from werkzeug.utils import secure_filename
import os

from models.blip import blip_decoder

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 384

model_url_art = 'static/checkpoint_best_art.pth'
model_art = blip_decoder(pretrained=model_url_art, image_size=image_size, vit='base').to(device)
model_art.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def generate_caption():

    uploaded_file = request.files['image']
    filename = secure_filename(uploaded_file.filename)
    filepath = os.path.join('static/', filename)
    uploaded_file.save(filepath)
    img = Image.open(uploaded_file)
    image_path = os.path.join('static', filename)

    image_size = 384

    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        caption_art = model_art.generate(img, sample=True, top_p=0.9, max_length=20, min_length=5)
        caption_text_art = 'Caption generated with model finetuned on Art dataset: ' + caption_art[0]
        
    return render_template('index.html', caption_text_art=caption_text_art, image_path=image_path, image_name=filename)

if __name__ == '__main__':
    port = 5002
    app.run(debug=True, port=port)