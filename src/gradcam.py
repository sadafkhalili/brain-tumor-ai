import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from torchvision import models
from torchvision import transforms


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


target_layer = model.layer4[-1]


activations = None
gradients = None


def forward_hook(module, input, output):

    global activations

    activations = output


def backward_hook(module, grad_input, grad_output):

    global gradients

    gradients = grad_output[0]


target_layer.register_forward_hook(
    forward_hook
)

target_layer.register_full_backward_hook(
    backward_hook
)


image_path = (
    "data/test/brain_glioma/brain_glioma_0002.jpg"
)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


image = Image.open(
    image_path
).convert("RGB")


input_tensor = transform(
    image
).unsqueeze(0)


output = model(
    input_tensor
)


predicted_class = output.argmax()


model.zero_grad()

output[
    0,
    predicted_class
].backward()


pooled_gradients = torch.mean(
    gradients,
    dim=[0, 2, 3]
)


for i in range(
    activations.shape[1]
):

    activations[
        :,
        i,
        :,
        :
    ] *= pooled_gradients[i]


heatmap = torch.mean(
    activations,
    dim=1
).squeeze()


heatmap = np.maximum(
    heatmap.detach().numpy(),
    0
)


heatmap /= np.max(
    heatmap
)


image_cv = cv2.imread(
    image_path
)


heatmap = cv2.resize(
    heatmap,
    (
        image_cv.shape[1],
        image_cv.shape[0]
    )
)


heatmap = np.uint8(
    255 * heatmap
)


heatmap = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)


superimposed_img = (
    heatmap * 0.4
    + image_cv
)


cv2.imwrite(
    "results/gradcam_result.jpg",
    superimposed_img
)


plt.imshow(
    cv2.cvtColor(
        superimposed_img.astype(
            np.uint8
        ),
        cv2.COLOR_BGR2RGB
    )
)

plt.axis("off")

plt.show()


print(
    "Grad-CAM Saved:"
)

print(
    "results/gradcam_result.jpg"
)