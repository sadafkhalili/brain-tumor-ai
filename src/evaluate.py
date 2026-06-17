import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms
from torchvision import models


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


test_dataset = datasets.ImageFolder(
    root="data/test",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)


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


correct = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


accuracy = 100 * correct / total

print(
    f"Test Accuracy: {accuracy:.2f}%"
)