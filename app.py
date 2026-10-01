import io
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow requests from Netlify and SoloLearn sandboxes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load lightweight pre-trained PyTorch model
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.eval()

# Image Preprocessing Pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

categories = weights.meta["categories"]

@app.get("/")
def home():
    return {"status": "PyTorch API is online"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Process image through PyTorch
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    top_prob, top_catid = torch.topk(probabilities, 1)
    
    label = categories[top_catid[0].item()]
    confidence = round(top_prob[0].item() * 100, 2)
    
    return {"label": label, "confidence": f"{confidence}%"}