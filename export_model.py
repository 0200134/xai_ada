from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class AiAdaNet(nn.Module):
    """Small exportable network used by the local flight simulator."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.fc(x)


def export_model(output_path: Optional[Path] = None) -> Path:
    torch.manual_seed(0)
    model = AiAdaNet().eval()
    dummy_input = torch.zeros(1, 4)
    destination = output_path or Path(__file__).resolve().with_name("ai_ada_model.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        str(destination),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["pilot_input"],
        output_names=["ai_ada_decision"],
        dynamic_axes={
            "pilot_input": {0: "batch_size"},
            "ai_ada_decision": {0: "batch_size"},
        },
    )
    return destination


if __name__ == "__main__":
    print(f"Successfully generated and exported: {export_model()}")
