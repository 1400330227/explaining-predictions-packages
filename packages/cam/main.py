import argparse
import cv2
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision import models

from cam.grad_cam import GradCAM


def get_net(net_name, weight_path=None):
    """
    It gets the model based on the net name.
    :param net_name:
    :param weight_path:
    :return:
    """
    pretrain = weight_path is None  # 没有指定权重路径，则加载默认的预训练权重
    if net_name in ['vgg', 'vgg16']:
        net = models.vgg16(pretrained=pretrain)
    elif net_name == 'vgg19':
        net = models.vgg19(pretrained=pretrain)
    elif net_name in ['resnet', 'resnet50']:
        net = models.resnet50(pretrained=pretrain)
    elif net_name == 'resnet101':
        net = models.resnet101(pretrained=pretrain)
    elif net_name in ['densenet', 'densenet121']:
        net = models.densenet121(pretrained=pretrain)
    elif net_name in ['inception']:
        net = models.inception_v3(pretrained=pretrain)
    elif net_name in ['mobilenet_v2']:
        net = models.mobilenet_v2(pretrained=pretrain)
    elif net_name in ['shufflenet_v2']:
        net = models.shufflenet_v2_x1_0(pretrained=pretrain)
    else:
        raise ValueError('invalid network name:{}'.format(net_name))
    print(net.eval())
    return net


def get_last_conv_name(net):
    """
    It gets the last convolution layer name of the net
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


def prepare_input(image_file):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    img_pil = Image.open(image_file)
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))  # add the batch value
    return img_variable, img_tensor


def main(args):
    img = cv2.imread(args.image_path)
    height, width, _ = img.shape

    img_variable, img_tensor = prepare_input(args.image_path)

    net = get_net(args.network, args.weight_path)

    layer_name = get_last_conv_name(net) if args.layer_name is None else args.layer_name
    grad_cam = GradCAM(net, layer_name)
    mask = grad_cam(img_variable, args.class_id)
    heatmap = cv2.applyColorMap(cv2.resize(mask, (width, height)), cv2.COLORMAP_JET)
    cv2.imwrite('CAM_heatmap.jpg', heatmap)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite('CAM.jpg', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='vgg16')
    parser.add_argument('--image_path', type=str, default='test.jpg')
    parser.add_argument('--weight-path', type=str, default=None)
    parser.add_argument('--layer-name', type=str, default=None, help='last convolutional layer name')
    parser.add_argument('--class-id', type=int, default=None, help='class id')

    arguments = parser.parse_args()
    main(arguments)
