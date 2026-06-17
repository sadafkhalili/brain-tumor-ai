import torch
import torch.nn as nn

from PIL import Image

from torchvision import transforms
from torchvision import models


classes = [
    "brain_glioma",
    "brain_menin",
    "brain_tumor"
]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


model = models.resnet18()

model.fc = nn.Linear(
    model.fc.in_features,
    3
)

model.load_state_dict(
    torch.load(
        "models/brain_tumor_model.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()


image_path = "sample.jpg"

image = Image.open(
    image_path
).convert("RGB")

image = transform(image)

image = image.unsqueeze(0)


with torch.no_grad():

    output = model(image)

    _, predicted = torch.max(
        output,
        1
    )


print(
    "Prediction:",
    classes[predicted.item()]
)