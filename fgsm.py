import torch
import torch.nn.functional as F

class FGSM:
    """
    Fast Gradient Sign Method Attack
    """

    def __init__(self, model, epsilon=0.1):
        self.model = model
        self.epsilon = epsilon

    def attack(self, image, label):
        """
        Generates adversarial image using FGSM
        """
        image.requires_grad = True

        output = self.model(image)
        loss = F.nll_loss(output, label)

        self.model.zero_grad()
        loss.backward()

        data_grad = image.grad.data
        sign_data_grad = data_grad.sign()

        perturbed_image = image + self.epsilon * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

        return perturbed_image
