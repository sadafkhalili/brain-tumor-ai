import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
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


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def predict_image():

    try:

        print("Button Clicked")

        file_path = filedialog.askopenfilename()

        print("Selected:", file_path)

        if not file_path:
            return

        image = Image.open(file_path).convert("RGB")

        print("Image Loaded")

        image = transform(image)

        image = image.unsqueeze(0)

        with torch.no_grad():

            output = model(image)

            probabilities = torch.softmax(
                output,
                dim=1
            )

            confidence, predicted = torch.max(
                probabilities,
                1
            )

        print("Prediction Done")

        print("Class:", classes[predicted.item()])
        print("Confidence:", confidence.item() * 100)
        messagebox.showinfo(
            "Prediction Result",
           f"Prediction: {classes[predicted.item()]}\n\nConfidence: {confidence.item()*100:.2f}%"
         )


    except Exception as e:

        print("ERROR:", e)

root = tk.Tk()
root.lift()
root.attributes("-topmost", True)

root.title(
    "Brain Tumor Classifier"
)

root.geometry("500x300")


title_label = tk.Label(
    root,
    text="Brain Tumor Classification",
    font=("Arial", 18)
)

title_label.pack(pady=20)


button = tk.Button(
    root,
    text="Select MRI Image",
    command=predict_image
)

button.pack(pady=20)


result_label = tk.Label(
    root,
    text="Prediction:",
    font=("Arial", 18)
)

result_label.pack(pady=10)


confidence_label = tk.Label(
    root,
    text="Confidence:",
    font=("Arial", 18)
)
confidence_label.pack(pady=10)


root.mainloop()