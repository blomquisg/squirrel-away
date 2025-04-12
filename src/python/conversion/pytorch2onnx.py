import torch
import torchvision.models as models

import config as cfg

config = cfg.Config()

model = models.mobilenet_v2()
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load("experiments/trained_20250302_183521/model", map_location=torch.device("cpu")))


model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(model, dummy_input, "experiments/trained_20250302_183521/squirrel_model.onnx", opset_version=11)