import torch
from torchvision import transforms
from PIL import Image

def load_anime_model(style_name='face_paint_512_v2'):
    model = torch.hub.load(
        'bryandlee/animegan2-pytorch:main',
        'generator',
        pretrained=style_name
    )
    model.eval()
    return model

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.unsqueeze(0))
    ])
    return transform(image)

def postprocess_tensor(tensor):
    tensor = tensor.squeeze().clamp(0, 1)
    return transforms.ToPILImage()(tensor)
