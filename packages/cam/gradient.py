import torch
from torch import nn, Tensor

from packages.cam.core import _CAM


class _GradCAM(_CAM):
    def __init__(self, model: nn.Module, target_layer) -> None:
        super(model, target_layer).__init__()

    def _backpropagation(self, scores: Tensor, class_idx, retain_graph=False):
        if isinstance(class_idx, int):
            loss = scores[:, class_idx].sum()
        else:
            loss = scores.gather(1, torch.tensor(class_idx, device=scores.device).view(-1, 1)).sum()
        self.model.zero_grad()
        loss.backward(retain_graph=retain_graph)


class GradCAM(_CAM):
    def __init__(self, model: nn.Module, target_layer) -> None:
        super(model, target_layer).__init__()

    def _get_weights(self, class_idx, scores: Tensor):
        self._backpropagation(scores, class_idx)


class SmoothGradCAMpp(_GradCAM):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)

    # def _get_weights(self, class_idx):
