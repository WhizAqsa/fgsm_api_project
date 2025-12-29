## Adversarial Attack Demo with FGSM

This project demonstrates adversarial attacks on neural networks using the Fast Gradient Sign Method (FGSM). A trained MNIST classifier is used to show how small, carefully crafted perturbations can cause a model to misclassify an image.

### 🎯 Project Overview
The backend is implemented using FastAPI + PyTorch, exposing an API endpoint that:

- Takes an image as input
- Applies FGSM with a configurable epsilon value
- Returns predictions, attack success status

### 🛠 Technology Stack

- FastAPI — REST API framework
- PyTorch — Model & FGSM implementation
- Torchvision — Image transformations
- Uvicorn — ASGI server
- PIL — Image processing

### 🚀 How to Run Locally (Backend)
1. Clone the Repository

- git clone `<your-repo-url>`
- cd `<project-folder>`

2. Install Dependencies

- pip install -r requirements.txt
- pip install torch torchvision

- Note: torch and torchvision are installed separately due to system‑specific builds.

4. Train the MNIST model (if not already trained)
   - `python3 train_mnist.py`

5. Run the FastAPI server
   - `uvicorn main:app --reload`

6. The backend will be available at:

- API: http://127.0.0.1:8000/attack
- Swagger UI: http://127.0.0.1:8000/docs

### 📡 API Endpoint Details
- POST /attack
- Inputs (multipart/form-data):
  - file: MNIST image (JPG/PNG)
  - epsilon: perturbation strength (float)

- Response Example:
  - {
      "clean_prediction": 9,
      "adversarial_prediction": 3,
      "attack_success": true,
      "attack_strength_percent": 18.91,
      "adversarial_image_base64": "<base64_string>"
    }

### What is FGSM?

The Fast Gradient Sign Method (FGSM) is an adversarial attack technique that slightly alters an input image in order to mislead a neural network. It works by calculating the gradient of the loss with respect to the input pixels and then modifying the image in the direction that increases the loss.

The modification is scaled by a parameter called epsilon (ε), which controls how strong the perturbation is. Even though the perturbation is often visually imperceptible to humans, it can cause a well‑trained model to make incorrect predictions.

FGSM is computationally efficient and is widely used to evaluate model robustness against adversarial inputs.

### 📊 Observations & Results
1. Observed Trend:
Attack success percentage follows a non‑linear pattern as epsilon increases.

2. Low Epsilon (ε ≈ 0.1–0.2):
- Perturbations are very small
- Model predictions mostly remain unchanged
- Attack success rate is low

3. Medium Epsilon (ε ≈ 0.4–0.6):
- Perturbations become effective but still structured
- Model is most vulnerable
- Attack success rate reaches its peak

4. High Epsilon (ε > 0.6):
- Perturbations become too strong and noisy
- Adversarial patterns lose directionality
- Attack success rate starts to decrease

5. Summary:
- Initial low success → increases to an optimal point → then decreases
- Higher epsilon increases perturbation strength only up to an optimal threshold
- Very high epsilon values introduce visible noise and can reduce attack effectiveness

### 🎓 Conclusion

This project demonstrates how fragile neural networks can be when exposed to adversarial perturbations. FGSM provides a simple yet powerful method to test model robustness and highlights the importance of secure and robust AI systems. The observed trend where attack success peaks at moderate epsilon values and then declines suggests that there is an optimal perturbation strength for adversarial attacks on this MNIST classifier.
