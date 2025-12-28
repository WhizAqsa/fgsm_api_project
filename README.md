# FGSM Adversarial Attack API (FastAPI + PyTorch)

## Overview
This project implements the **Fast Gradient Sign Method (FGSM)** adversarial attack
and exposes it through a **REST API** using **FastAPI**. The system evaluates the
robustness of a pretrained **MNIST classification model** against adversarial examples.

FGSM is a gradient-based attack that introduces small but carefully crafted
perturbations to input images, causing deep learning models to misclassify them.

---

## Objectives
- Implement FGSM in PyTorch using a modular design
- Evaluate the robustness of a pretrained MNIST model
- Build a REST API to perform adversarial attacks on uploaded images
- Return predictions, attack success status, and adversarial images

---

## Project Structure
fgsm_api_project/
│
├── fgsm.py # FGSM attack class
├── model.py # MNIST CNN model
├── app_fgsm.py # FastAPI application
├── requirements.txt # Dependencies
├── outputs/ # Results & screenshots
└── README.md


---

## Fast Gradient Sign Method (FGSM)
FGSM generates adversarial examples using the following formula:

\[
x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y))
\]

Where:
- `x` is the original input image
- `y` is the true label
- `J` is the loss function
- `ε (epsilon)` controls the attack strength

---

## FGSM Implementation
The FGSM attack is implemented in `fgsm.py` using PyTorch’s automatic differentiation.
The attack computes the gradient of the loss with respect to the input image and
perturbs the image in the direction of the gradient sign.

---

## Model and Evaluation
- A pretrained CNN model is used for MNIST digit classification
- The model’s accuracy is evaluated before and after applying FGSM
- Increasing epsilon results in a noticeable drop in accuracy, demonstrating
  vulnerability to adversarial attacks

Only final results and screenshots are included, as required.

---

## FastAPI Endpoint

### POST `/attack`

**Input:**
- Image file (PNG or JPEG)
- Epsilon value (default = 0.1)

**Output (JSON):**
- Clean Prediction
- Adversarial Prediction
- Attack Success Status
- Base64-encoded adversarial image

---

## How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
