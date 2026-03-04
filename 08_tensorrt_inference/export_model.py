import torch
import torchvision.models as models

model = models.resnet18()
model.eval()

dummy = torch.randn(1,3,224,224)

torch.onnx.export(
    model,
    dummy,
    "resnet18.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

print("ONNX model exported")