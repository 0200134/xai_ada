import torch
import torch.nn as nn

# 1. AI-Ada 더미 신경망 모델 정의
# 입력: 4개 (pitch, roll, yaw, airspeed)
# 출력: 2개 (corrected_pitch, safety_flag)
class AiAdaNet(nn.Module):
    def __init__(self):
        super(AiAdaNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 2) # [보정된 피치, 안전 여부 (0 또는 1)]
        )

    def forward(self, x):
        return self.fc(x)

# 2. 모델 인스턴스 생성 및 가중치 초기화
model = AiAdaNet()
model.eval()

# 3. 더미 입력 데이터 (Batch Size: 1, Features: 4)
dummy_input = torch.randn(1, 4)

# 4. ONNX 포맷으로 내보내기 (Export)
onnx_path = "ai_ada_model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['pilot_input'],
    output_names=['ai_ada_decision'],
    dynamic_axes={'pilot_input': {0: 'batch_size'}, 'ai_ada_decision': {0: 'batch_size'}}
)

print(f"Successfully generated and exported: {onnx_path}")
