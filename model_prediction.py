import torch
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import sys

# ======== CONFIGURATION ========
image_path = sys.argv[1]
model_weights_path = "plant_model.pth"
num_classes = 15
class_names = ImageFolder("dataset_split/train").classes  # Loads class names from folders

# ======== IMAGE TRANSFORM (MUST MATCH VALIDATION SETUP) ========
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])

# ======== LOAD MODEL ========
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
model.eval()

# ======== LOAD AND TRANSFORM IMAGE ========
image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# ======== INFERENCE ========
with torch.no_grad():
    outputs = model(input_tensor)
    probs = F.softmax(outputs, dim=1)
    predicted_class_index = torch.argmax(probs, dim=1).item()
    predicted_class = class_names[predicted_class_index]
    confidence = probs[0][predicted_class_index].item()

# ======== OUTPUT ========
print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence*100:.2f}%")
