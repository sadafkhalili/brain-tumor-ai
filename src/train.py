import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms
from torchvision import models


# Transform

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# Datasets

train_dataset = datasets.ImageFolder(
    root="data/train",
    transform=transform
)

valid_dataset = datasets.ImageFolder(
    root="data/valid",
    transform=transform
)


# DataLoaders

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=16,
    shuffle=False
)


# Model

model = models.resnet18(
    weights=models.ResNet18_Weights.IMAGENET1K_V1
)

model.fc = nn.Linear(
    model.fc.in_features,
    3
)


# Loss

criterion = nn.CrossEntropyLoss()


# Optimizer

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# Early Stopping Settings

best_accuracy = 0

patience = 3
counter = 0

EPOCHS = 10


# Training Loop

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()


    # Validation

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in valid_loader:

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
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {running_loss:.4f} | "
        f"Val Accuracy: {accuracy:.2f}%"
    )


    # Save Best Model

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        counter = 0

        torch.save(
            model.state_dict(),
            "models/brain_tumor_model.pth"
        )

        print("Best Model Saved")

    else:

        counter += 1

        print(
            f"No Improvement ({counter}/{patience})"
        )


    # Early Stopping

    if counter >= patience:

        print(
            "Early Stopping Triggered"
        )

        break


print("Training Finished")
print(
    f"Best Accuracy: {best_accuracy:.2f}%"
)