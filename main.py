from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI(title="API de Détection des Maladies du Riz")

# 1. Définition des classes (dans le même ordre que lors de l'entraînement)
class_names = [
    'bacterial_leaf_blight', 
    'brown_spot', 
    'healthy', 
    'leaf_blast', 
    'leaf_scald', 
    'narrow_brown_spot'
]

# 2. Chargement du modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reconstruire l'architecture ResNet18
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

# Charger les poids entraînés
model.load_state_dict(torch.load("model_rice_disease.pth", map_location=device))
model = model.to(device)
model.eval()

# 3. Transformations pour l'image envoyée
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API de détection des maladies du riz ! Envoyez une image sur /predict"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Lecture de l'image reçue
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Prétraitement
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Prédiction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        conf, predicted = torch.max(probabilities, 0)
        
    predicted_class = class_names[predicted.item()]
    confidence = conf.item() * 100
    
    return {
        "prediction": predicted_class,
        "confidence": f"{confidence:.2f}%"
    }