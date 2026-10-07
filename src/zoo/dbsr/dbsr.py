
import torch
import torch.nn as nn

from ...core import register

__all__ = [
    "DBSR",
]



@register()
class DBSR(nn.Module):
    __inject__ = [
        "backbone",
        "encoder",
        "decoder",
        "VisualClassifier",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        VisualClassifier: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.VisualClassifier = VisualClassifier

    def forward(self, x, targets=None):
        images = x
        backbone_feats = self.backbone(images)
        enc_feats = self.encoder(backbone_feats)
        outputs = self.decoder(enc_feats, targets)
        outputs1 = self.VisualClassifier(enc_feats, outputs, targets)
        outputs.pop("decoder_query_feats", None)

        return outputs, outputs1

    @torch.no_grad()
    def sample(self, x, targets=None):
        images = x
        backbone_feats = self.backbone(images)
        enc_feats = self.encoder(backbone_feats)
        outputs = self.decoder(enc_feats)
        outputs = self.VisualClassifier(enc_feats, outputs, targets)

        return outputs, enc_feats

    def get_losses(self, outputs, **kwargs):
        if outputs is None:
            return {}
        return self.VisualClassifier.get_losses(outputs, **kwargs)

    def deploy(
        self,
    ):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self
