
import torch
import torchvision.models as models

# Load pre-trained CNN model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Print architecture summary
print(model)
