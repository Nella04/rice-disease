from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# Réduire les threads CPU pour limiter les ressources utilisées
torch.set_num_threads(1)

# API
app = FastAPI(title="API de Détection des Maladies du Riz")

# 1. Classes
class_names = [
    'bacterial_leaf_blight',
    'brown_spot',
    'healthy',
    'leaf_blast',
    'leaf_scald',
    'narrow_brown_spot'
]

# 2. Configuration CPU
device = torch.device("cpu")

# 3. Chargement du modèle
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))

state_dict = torch.load(
    "model_rice_disease.pth",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()

# 4. Transformation des images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# 5. Accueil
@app.get("/")
def home():
    return {
        "message": "API de détection des maladies du riz opérationnelle."
    }


# 6. Prédiction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Format d'image non accepté."
        )

    image_bytes = await file.read()

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            image = img.convert("RGB")
            image = transform(image).unsqueeze(0)

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Image invalide."
        )

    # Inférence sans calcul de gradients
    with torch.inference_mode():
        outputs = model(image)
        probabilities = torch.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, dim=0)

    return {
        "prediction": class_names[predicted.item()],
        "confidence": f"{confidence.item() * 100:.2f}%"
    }
```