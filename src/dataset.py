from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    root="data/train",
    transform=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

print("Classes:", dataset.classes)

print("Number of Images:", len(dataset))

images, labels = next(iter(dataloader))

print("Batch Shape:", images.shape)

print("Labels Shape:", labels.shape)