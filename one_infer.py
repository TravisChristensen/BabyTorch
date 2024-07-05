import torch
import string
from torchvision import transforms
from PIL import Image

from device import get_device
from networks import ConvNet


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


if __name__ == "__main__":
    label_map = {}
    for i, letter in enumerate(zip(list(string.ascii_lowercase), list(string.ascii_uppercase))):
        lower, upper = letter
        label_map[i] = f"{upper}/{lower}"
    ctx = get_device()
    model = ConvNet().to(ctx)
    model.load_state_dict(torch.load('convnet_emnist.pth'))
    model.eval()
    image = preprocess_image("k.png").to(ctx)
    with torch.no_grad():
        output = model(image)
        predicted_label = torch.argmax(output, dim=1)
        label_val = predicted_label.item()
        print(f'Predicted Label: {label_val}:{label_map[label_val]}')
