import cv2
import numpy as np

__all__ = ['GradCAM']

from sympy.physics.vector import gradient


class GradCAM(object):

    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.handlers = []
        self._register_hooks()
        self.feature = None
        self.gradient = None

    def _get_features_hook(self, module, input, output):
        self.feature = output
        print("feature shape:{}".format(output.size()))

    def _get_grads_hook(self, module, grad_in, grad_out):
        self.gradient = grad_out[0]
        print("gradient shape:{}".format(grad_out[0].size()))

    def _register_hooks(self):
        for (name, module) in self.model.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handler in self.handlers:
            handler.remove()

    def __call__(self, inputs, index):
        """
        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        self.model.zero_grad()
        output = self.model(inputs)
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]
        target.backward()

        bz, nc, h, w = self.feature.shape

        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        weight = np.mean(gradient, axis=(1, 2))  # [C]
        feature = self.feature[0].cpu().data.numpy()  # [C,H,W]

        # feature = feature.reshape(nc, h * w)

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        # cam = weight * feature
        cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)  # ReLU

        # 数值归一化
        cam -= np.min(cam)
        cam /= (np.max(cam) - np.min(cam))
        cam = np.uint8(255 * cam)
        cam = cv2.resize(cam, (224, 224))
        return cam
