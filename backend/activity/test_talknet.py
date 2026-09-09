import sys
import torch
from pathlib import Path

TALKNET_DIR = Path(__file__).resolve().parents[2] / "TalkNet-ASD"
sys.path.insert(0, str(TALKNET_DIR))

from talkNet import talkNet

print("Loading TalkNet...")
model = talkNet()
model_path = Path(__file__).resolve().parents[2] / "TalkNet-ASD" / "pretrain_TalkSet.model"
model.loadParameters(str(model_path))
model.eval()

talknet_model = model.model
device = next(talknet_model.parameters()).device
print(f"Model loaded on {device}")

test_video = torch.rand(1, 50, 112, 112).to(device)
test_audio = torch.rand(1, 200, 13).to(device)

with torch.no_grad():
    v = talknet_model.forward_visual_frontend(test_video)
    a = talknet_model.forward_audio_frontend(test_audio)
    a, v = talknet_model.forward_cross_attention(a, v)
    out = talknet_model.forward_audio_visual_backend(a, v)
    print("Raw logits:", out)
    print("Softmax probs:", torch.softmax(out, dim=-1))
    print("outs_av shape:", out.shape)
    pred_score = model.lossAV.forward(out, labels=None)

    print("pred_score:", pred_score)