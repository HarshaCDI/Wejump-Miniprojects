import torch
from torchvision import transforms
from PIL import Image

def load_model(style_name='paprika'):
    model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained=style_name)
    model.eval()
    return model

def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Unsqueeze(0)
    ])
    return tf(img)

def postprocess(tensor):
    tensor = tensor.squeeze().clamp(0, 1)
    return transforms.ToPILImage()(tensor)
