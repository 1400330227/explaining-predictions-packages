from abc import abstractmethod

from torch import nn

__all__ = ['_CAM']


class _CAM:
    def __init__(self, model: nn.Module, target_layer=None):
        self.model = model

        if isinstance(target_layer, str):
            target_names = [target_layer]
        elif isinstance(target_layer, nn.Module):
            target_names = [target_layer.__class__.__name__]

    @abstractmethod
    def _get_weights(self, class_idx):
        raise NotImplementedError
