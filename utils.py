import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

# ImageFolder sorts class folders alphabetically, so this order matches training.
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def get_transform(train: bool = False):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def build_model(num_classes: int = len(CLASSES), pretrained: bool = True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class GradCAM:
    """Highlights the image regions that influenced the prediction."""

    def __init__(self, model, target_layer):
        self.model = model
        self.acts = None
        self.grads = None
        target_layer.register_forward_hook(self._save)

    def _save(self, module, inp, out):
        self.acts = out.detach()
        out.register_hook(lambda g: setattr(self, "grads", g.detach()))

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        logits = self.model(x)
        probs = logits.softmax(1).detach().squeeze().cpu().numpy()
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        logits[0, class_idx].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.acts).sum(1)).squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam, probs, class_idx


def overlay_heatmap(pil_img, cam):
    img = np.array(pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (0.55 * img + 0.45 * heat).astype(np.uint8)
