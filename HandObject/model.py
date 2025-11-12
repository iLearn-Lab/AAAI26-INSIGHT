import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils as nn_utils
from typing import Tuple

class RegularizedMLP(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=2048):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn_utils.weight_norm(nn.Linear(in_dim, hidden_dim))
        self.act1 = nn.GELU()
        self.fc2 = nn_utils.weight_norm(nn.Linear(hidden_dim, in_dim))
        self.act2 = nn.GELU()
        self.norm = nn.LayerNorm(in_dim)
        self._init_weights()

    def _init_weights(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.fc1(x)
        out = self.act1(out)
        out = self.fc2(out)
        out = self.act2(out)
        out = out + identity
        out = self.norm(out)
        return out

class ActionRecognitionModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 1408,
        mlp_hidden_dim: int = 2048,
        mlp_output_dim: int = 256,
        transformer_layers: int = 4,
        n_heads: int = 8,
        transformer_hidden_dim: int = 2048,
        verb_num_classes: int = 117,
        noun_num_classes: int = 521,
    ):
        super().__init__()
        self.frame_stage1 = self._make_lowrank_layer(1408, 2048)
        self.frame_norm1 = nn.LayerNorm(2048)
        self.frame_gate1 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Linear(512, 2048),
            nn.Sigmoid()
        )
        self.frame_stage2 = nn.Linear(2048, 256)
        self.mask_stage1 = self._make_lowrank_layer(1408, 2048)
        self.mask_norm1 = nn.LayerNorm(2048)
        self.mask_gate1 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Linear(512, 2048),
            nn.Sigmoid()
        )
        self.mask_stage2 = nn.Linear(2048, 256)
        self.frame_mlp = RegularizedMLP(in_dim=256, hidden_dim=mlp_hidden_dim)
        self.mask_mlp = RegularizedMLP(in_dim=256, hidden_dim=mlp_hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=n_heads,
            dim_feedforward=transformer_hidden_dim,
            activation='relu',
            batch_first=True,
            dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.verb_head = nn.Linear(512, verb_num_classes)
        self.noun_head = nn.Linear(512, noun_num_classes)
        self._init_weights()

    def _make_lowrank_layer(self, in_dim, out_dim):
        return nn.Linear(in_dim, out_dim)

    def _init_weights(self):
        init.xavier_uniform_(self.verb_head.weight)
        init.xavier_uniform_(self.noun_head.weight)
        if self.verb_head.bias is not None:
            init.zeros_(self.verb_head.bias)
        if self.noun_head.bias is not None:
            init.zeros_(self.noun_head.bias)

    def forward(self, frame_features: torch.Tensor, mask_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = frame_features.size(0)
        if frame_features.dim() == 4:
            frame_features = frame_features.squeeze(1)
        if mask_features.dim() == 4:
            mask_features = mask_features.squeeze(1)
        frame_out = self.frame_stage1(frame_features)
        frame_out = self.frame_norm1(frame_out)
        frame_gate = self.frame_gate1(frame_out)
        frame_out = frame_out * frame_gate
        frame_out = self.frame_stage2(frame_out)
        mask_out = self.mask_stage1(mask_features)
        mask_out = self.mask_norm1(mask_out)
        mask_gate = self.mask_gate1(mask_out)
        mask_out = mask_out * mask_gate
        mask_out = self.mask_stage2(mask_out)
        frame_out = self.frame_mlp(frame_out)
        mask_out = self.mask_mlp(mask_out)
        combined = torch.cat([frame_out, mask_out], dim=-1)
        cls_tokens = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls_tokens, combined], dim=1)
        x = self.transformer(x)
        cls_output = x[:, 0, :]
        verb_logits = self.verb_head(cls_output)
        noun_logits = self.noun_head(cls_output)
        return verb_logits, noun_logits
