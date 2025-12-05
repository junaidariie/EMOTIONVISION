import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torchvision import models

class FaceExpressionResnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.model = models.resnet18(weights="DEFAULT")

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                for param in module.parameters():
                    param.requires_grad = True

        self.model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.model.fc.in_features, num_classes)
        )


    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 5
model = FaceExpressionResnet(num_classes).to(device)

model.load_state_dict(torch.load("artifacts/FaceExpressionResnet_18.pth", map_location=device))
model.eval()

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = [
    "Angry",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
]

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = test_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs).item()
        confidence = probs[0][pred_idx].item()

    return {'emotion' : class_names[pred_idx], "confidence" : confidence}

"""img_path = "10069404.png"

img = predict_image(img_path)
print(img)
"""