from fastapi import FastAPI
from pydantic import BaseModel
import torch
import base64
import io
from PIL import Image
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware

from model import MultimodalFakeNewsDetectionModel
from model import JointTextImageModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "saved_model/fake_news_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hparams = {
    "embedding_dim": 768,
    "num_classes": 2
}

model = MultimodalFakeNewsDetectionModel(hparams)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

text_embedder = SentenceTransformer("all-mpnet-base-v2")

image_transform = JointTextImageModel.build_image_transform()

class NewsRequest(BaseModel):
    text: str
    image: str  # base64 encoded image


@app.post("/predict")
def predict(news: NewsRequest):

    # TEXT
    text_embedding = text_embedder.encode(news.text, convert_to_tensor=True)
    text_embedding = text_embedding.unsqueeze(0).to(device)

    # IMAGE
    if news.image == "" or news.image is None:
        # No image provided → use dummy image
        image_tensor = torch.ones((1,3,224,224)).to(device) * 0.5
    else:
        image_data = base64.b64decode(news.image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_tensor = image_transform(image).unsqueeze(0).to(device)
        label = torch.zeros(1).long().to(device)

    # create dummy label
    label = torch.zeros(1).long().to(device)

    with torch.no_grad():
        preds, _ = model(text_embedding, image_tensor, label)

        probabilities = torch.softmax(preds, dim=1)
        confidence = probabilities.max().item()

        prediction = torch.argmax(preds, dim=1).item()

    return {
        "prediction": "Fake News" if prediction == 1 else "Real News",
        "confidence": round(confidence, 3)
    }