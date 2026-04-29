import torch
import torch.nn as nn
from transformers import AutoModel

MODEL_NAME  = "distilbert-base-uncased"
NUM_CLASSES = 4
DROPOUT     = 0.3


class DealStageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls    = output.last_hidden_state[:, 0, :]
        return self.classifier(cls)


def load_model(checkpoint_path=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model  = DealStageClassifier().to(device)
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[model] Loaded from {checkpoint_path}")
    print(f"[model] Ready on {device}")
    return model, device