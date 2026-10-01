import io
import base64
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

# LangChain & LangGraph Integrations
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# Choose your provider (uncomment preferred model)
from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic

app = FastAPI(title="LangChain Tool-Calling Vision Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 1. PyTorch Setup & Global State
# ---------------------------------------------------------
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

# Temporary global buffer to pass image tensor into tool
current_image_buffer: Optional[Image.Image] = None


# ---------------------------------------------------------
# 2. Define LangChain Custom Tool
# ---------------------------------------------------------
@tool
def run_pytorch_resnet_classifier() -> str:
    """Runs a local PyTorch ResNet-18 model on the currently uploaded image.
    Returns the top 3 visual predictions with confidence percentages.
    """
    global current_image_buffer
    if current_image_buffer is None:
        return "Error: No image loaded into the vision buffer."

    tensor = transform(current_image_buffer).unsqueeze(0)
    with torch.no_grad():
        outputs = vision_model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    top_prob, top_catid = torch.topk(probabilities, 3)
    results = []
    for i in range(3):
        label = categories[top_catid[i].item()]
        conf = round(top_prob[i].item() * 100, 2)
        results.append(f"{label} ({conf}%)")

    return f"PyTorch Vision Output: {', '.join(results)}"


# ---------------------------------------------------------
# 3. Instantiate Model and Agent Loop
# ---------------------------------------------------------
# OpenAI Configuration
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Alternative Anthropic Claude Configuration:
# llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

tools = [run_pytorch_resnet_classifier]

system_prompt = (
    "You are an expert Vision AI Agent. Your goal is to identify objects in images "
    "by combining your visual reasoning with specialized computer vision tools. "
    "When presented with an image, ALWAYS call the `run_pytorch_resnet_classifier` tool "
    "to obtain quantitative predictions before synthesizing your final analysis."
)

agent_executor = create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_prompt
)


# ---------------------------------------------------------
# 4. API Endpoints
# ---------------------------------------------------------
class AgentResponse(BaseModel):
    status: str
    thought_process: List[str]
    final_answer: str


@app.get("/")
def check_health():
    return {"status": "LangChain Vision Agent Core Online"}


@app.post("/agent/analyze", response_model=AgentResponse)
async def run_agent(file: UploadFile = File(...)):
    global current_image_buffer
    thought_process = []

    # Read and buffer image
    contents = await file.read()
    try:
        current_image_buffer = Image.open(io.BytesIO(contents)).convert("RGB")
        base64_image = base64.b64encode(contents).decode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file format.")

    # Format multi-modal message for LLM
    input_message = HumanMessage(
        content=[
            {"type": "text", "text": "Analyze this image using your local PyTorch classifier tool and explain what you see."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        ]
    )

    thought_process.append("User query & image received by Agent.")

    try:
        # Execute LangGraph Tool-Calling Agent Loop
        events = agent_executor.stream(
            {"messages": [input_message]},
            stream_mode="values"
        )

        final_answer = ""
        for event in events:
            messages = event.get("messages", [])
            if messages:
                latest = messages[-1]
                # Log tool calls or standard messages
                if hasattr(latest, "tool_calls") and latest.tool_calls:
                    for tc in latest.tool_calls:
                        thought_process.append(f"Agent Decision: Calling Tool `{tc['name']}`")
                elif latest.type == "tool":
                    thought_process.append(f"Tool Result: {latest.content}")
                elif latest.type == "ai" and latest.content:
                    final_answer = latest.content

        return AgentResponse(
            status="success",
            thought_process=thought_process,
            final_answer=final_answer
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
