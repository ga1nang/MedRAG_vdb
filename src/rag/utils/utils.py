import base64
from PIL import Image

def encode_image(image_path):
    image = Image.open(image_path)
    return image