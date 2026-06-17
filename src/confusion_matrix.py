import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms
from torchvision import models

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt


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


all_labels = []
all_predictions = []


with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        all_labels.extend(labels.numpy())
        all_predictions.extend(predicted.numpy())


cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:\n")
print(cm)

print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=test_dataset.classes
    )
)


plt.figure(figsize=(6, 6))

plt.imshow(cm)

plt.colorbar()

plt.xticks(
    range(len(test_dataset.classes)),
    test_dataset.classes,
    rotation=45
)

plt.yticks(
    range(len(test_dataset.classes)),
    test_dataset.classes
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Brain Tumor Confusion Matrix")

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(
            j,
            i,
            str(cm[i][j]),
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png"
)

print(
    "\nSaved: results/confusion_matrix.png"
)