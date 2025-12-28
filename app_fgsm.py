from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import base64
import io
import os

from model import MNISTModel
from fgsm import FGSM
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = MNISTModel().to(device)
model.load_state_dict(torch.load("mnist_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])


@app.post("/attack")
async def attack_image(
    file: UploadFile = File(...),
    epsilon: float = 0.1
):
    image = Image.open(file.file)
    image = transform(image).unsqueeze(0).to(device)

    # Clean prediction and softmax probabilities
    output = model(image)
    clean_probs = F.softmax(output, dim=1)
    clean_pred = output.argmax(dim=1).item()
    clean_confidence = clean_probs.max().item()  # Confidence of predicted class

    label = torch.tensor([clean_pred]).to(device)

    # FGSM attack
    fgsm = FGSM(model, epsilon)
    adv_image = fgsm.attack(image, label)

    # Adversarial prediction and softmax
    adv_output = model(adv_image)
    adv_probs = F.softmax(adv_output, dim=1)
    adv_pred = adv_output.argmax(dim=1).item()
    adv_confidence = adv_probs.max().item()  # Confidence of predicted class

    # Determine if attack succeeded
    attack_success = clean_pred != adv_pred

    # Compute attack strength percentage (confidence drop) only if attack succeeded
    attack_strength_percent = round(abs(clean_confidence - adv_confidence) * 100, 2) if attack_success else 0.0

    # Convert adversarial image to Base64
    adv_pil = transforms.ToPILImage()(adv_image.squeeze().cpu())

    # --- Save to output folder ---
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    adv_pil.save(os.path.join(output_dir, "adv_image.png"))

    # Convert adversarial image to Base64
    buffer = io.BytesIO()
    adv_pil.save(buffer, format="PNG")
    adv_base64 = base64.b64encode(buffer.getvalue()).decode()

    return {
        "clean_prediction": clean_pred,
        "adversarial_prediction": adv_pred,
        "attack_success": attack_success,
        "attack_strength_percent": attack_strength_percent,
        "adversarial_image_base64": adv_base64
    }
