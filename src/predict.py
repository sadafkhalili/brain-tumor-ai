import torch
import torch.nn as nn

from torchvision import models
from torchvision import transforms

from PIL import Image


CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "Tumor"
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


image = Image.open(
    "sample.jpg"
).convert("RGB")

image = transform(image)

image = image.unsqueeze(0)


with torch.no_grad():

    output = model(image)

    prediction = torch.argmax(
        output,
        dim=1
    )

    predicted_class = CLASS_NAMES[
        prediction.item()
    ]


print(
    "Prediction:",
    predicted_class
)