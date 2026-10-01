import io
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

app = FastAPI(title="Vision AI Agent API")

# Enable CORS for Netlify and SoloLearn sandbox environments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tool 1: Vision Model Tool
weights = ResNet18_Weights.DEFAULT
vision_model = resnet18(weights=weights)
vision_model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
categories = weights.meta["categories"]

def tool_classify_image(image: Image.Image) -> dict:
    """Executes computer vision inference on the provided image."""
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = vision_model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    top_prob, top_catid = torch.topk(probabilities, 3)
    results = []
    for i in range(3):
        results.append({
            "label": categories[top_catid[i].item()],
            "confidence": round(top_prob[i].item() * 100, 2)
        })
    return {"top_predictions": results}

# Agent Execution Loop
class AgentResponse(BaseModel):
    status: str
    thought_process: list[str]
    primary_classification: str
    confidence: float
    agent_summary: str

@app.get("/")
def check_health():
    return {"status": "AI Agent Core Online"}

@app.post("/agent/analyze", response_model=AgentResponse)
async def run_agent(file: UploadFile = File(...)):
    thoughts = []
    
    # Step 1: Agent receives and validates input
    thoughts.append("Agent Goal: Analyze uploaded image and perform structured classification.")
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        thoughts.append(f"Received image of size {image.size}. Image successfully decoded.")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file uploaded.")

    # Step 2: Agent decides to call the PyTorch Vision Tool
    thoughts.append("Action: Executing PyTorch ResNet18 Computer Vision Tool...")
    vision_results = tool_classify_image(image)
    top_pred = vision_results["top_predictions"][0]
    thoughts.append(f"Tool Output: Detected '{top_pred['label']}' with {top_pred['confidence']}% confidence.")

    # Step 3: Agent synthesizes result and generates reasoning summary
    if top_pred['confidence'] > 70.0:
        summary = f"High confidence match! The target object is identified as a {top_pred['label']}."
    else:
        alt_label = vision_results["top_predictions"][1]["label"]
        summary = f"Moderate confidence match for {top_pred['label']}. Secondary candidate: {alt_label}."
    
    thoughts.append("Final Synthesis: Formatting structured agent payload for client.")

    return AgentResponse(
        status="success",
        thought_process=thoughts,
        primary_classification=top_pred["label"],
        confidence=top_pred["confidence"],
        agent_summary=summary
    )
