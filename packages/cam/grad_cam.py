import cv2
import numpy as np

__all__ = ['GradCAM']

import torch


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

    # def __call__(self, inputs, index):
    #     """
    #     :param inputs: [1,3,H,W]
    #     :param index: class id
    #     :return:
    #     """
    #     self.model.zero_grad()
    #     output = self.model(inputs)
    #     if index is None:
    #         index = np.argmax(output.cpu().data.numpy())
    #     target = output[0][index]
    #     target.backward()
    #
    #     bz, nc, h, w = self.feature.shape
    #
    #     gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
    #     weight = np.mean(gradient, axis=(1, 2))  # [C]
    #     feature = self.feature[0].cpu().data.numpy()  # [C,H,W]
    #
    #     cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
    #     cam = np.sum(cam, axis=0)  # [H,W]
    #     cam = np.maximum(cam, 0)  # ReLU
    #
    #     # 数值归一化
    #     cam -= np.min(cam)
    #     cam /= (np.max(cam) - np.min(cam))
    #     cam = np.uint8(255 * cam)
    #     cam = cv2.resize(cam, (224, 224))
    #     return cam

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
        size_upsample = (256, 256)

        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        weight = np.mean(gradient, axis=(1, 2))  # [C]

        feature = self.feature.cpu().data.numpy()

        feature_conv = feature.reshape((nc, h * w))
        weight_softmax = weight

        cam = weight_softmax.dot(feature_conv)
        cam = cam.reshape((h, w))
        cam = cam - np.min(cam)

        cam_img = cam / (np.max(cam) - np.min(cam))
        cam_img = np.uint8(255 * cam_img)
        cam_img = cv2.resize(cam_img, size_upsample)
        return cam_img
