import numpy as np
import torch.nn.functional as F
import torch
from torch import nn
from torchvision import models

np.random.seed(1)
torch.manual_seed(1)

inception_v3 = models.inception_v3(pretrained=True)


class InceptionV3(nn.Module):
    def __init__(self, num_classes, n_attributes=0, expand_dim=0, aux_logits=True):
        super(InceptionV3, self).__init__()
        self.aux_logits = aux_logits
        self.n_attributes = n_attributes
        self.Conv2d_1a_3x3 = inception_v3.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception_v3.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception_v3.Conv2d_2b_3x3
        self.maxpool1 = inception_v3.maxpool1
        self.Conv2d_3b_1x1 = inception_v3.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception_v3.Conv2d_4a_3x3
        self.maxpool2 = inception_v3.maxpool2
        self.Mixed_5b = inception_v3.Mixed_5b
        self.Mixed_5c = inception_v3.Mixed_5c
        self.Mixed_5d = inception_v3.Mixed_5d
        self.Mixed_6a = inception_v3.Mixed_6a
        self.Mixed_6b = inception_v3.Mixed_6b
        self.Mixed_6c = inception_v3.Mixed_6c
        self.Mixed_6d = inception_v3.Mixed_6d
        self.Mixed_6e = inception_v3.Mixed_6e
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes, n_attributes=self.n_attributes, bottleneck=True,
                                          expand_dim=expand_dim, three_class=False, connect_CY=False)
        self.Mixed_7a = inception_v3.Mixed_7a
        self.Mixed_7b = inception_v3.Mixed_7b
        self.Mixed_7c = inception_v3.Mixed_7c
        self.all_fc = nn.ModuleList()
        self.avgpool = inception_v3.avgpool
        self.dropout = inception_v3.dropout

        for i in range(self.n_attributes):
            self.all_fc.append(FC(2048, 1, expand_dim))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):  # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)  # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)  # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)  # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)  # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)  # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # N x 192 x 35 x 35
        x = self.Mixed_5b(x)  # N x 256 x 35 x 35
        x = self.Mixed_5c(x)  # N x 288 x 35 x 35
        x = self.Mixed_5d(x)  # N x 288 x 35 x 35
        x = self.Mixed_6a(x)  # N x 768 x 17 x 17
        x = self.Mixed_6b(x)  # N x 768 x 17 x 17
        x = self.Mixed_6c(x)  # N x 768 x 17 x 17
        x = self.Mixed_6d(x)  # N x 768 x 17 x 17
        x = self.Mixed_6e(x)  # N x 768 x 17 x 17
        out_aux = None
        if self.training and self.aux_logits:
            out_aux = self.AuxLogits(x)  # N x 768 x 17 x 17
        x = self.Mixed_7a(x)  # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)  # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)  # N x 2048 x 8 x 8
        x = F.adaptive_avg_pool2d(x, (1, 1))  # N x 2048 x 1 x 1
        x = F.dropout(x, training=self.training)  # N x 2048 x 1 x 1
        x = x.view(x.size(0), -1)  # N x 2048
        out = []
        for fc in self.all_fc:
            out.append(fc(x))
        if self.training and self.aux_logits:
            return out, out_aux
        else:
            return out


class FC(nn.Module):

    def __init__(self, input_dim, output_dim, expand_dim, stddev=None):
        super(FC, self).__init__()
        self.expand_dim = expand_dim
        if self.expand_dim > 0:
            self.relu = nn.ReLU()
            self.fc_new = nn.Linear(input_dim, expand_dim)
            self.fc = nn.Linear(expand_dim, output_dim)
        else:
            self.fc = nn.Linear(input_dim, output_dim)
        if stddev:
            self.fc.stddev = stddev
            if expand_dim > 0:
                self.fc_new.stddev = stddev

    def forward(self, x):
        if self.expand_dim > 0:
            x = self.fc_new(x)
            x = self.relu(x)
        x = self.fc(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, expand_dim):
        super(MLP, self).__init__()
        self.expand_dim = expand_dim
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.linear(x)
        return x


class End2EndModel(nn.Module):
    def __init__(self, model1, model2, use_relu=False, use_sigmoid=False, n_class_attr=2):
        super(End2EndModel, self).__init__()
        self.first_model = model1
        self.sec_model = model2
        self.use_relu = use_relu
        self.use_sigmoid = use_sigmoid

    def forward_stage2(self, stage1_out):
        if self.use_relu:
            attr_outputs = [nn.ReLU(inplace=True)(o) for o in stage1_out]
        elif self.use_sigmoid:
            attr_outputs = [nn.Sigmoid()(o) for o in stage1_out]
        else:
            attr_outputs = stage1_out

        stage2_inputs = attr_outputs
        stage2_inputs = torch.cat(stage2_inputs, dim=1)
        all_out = [self.sec_model(stage2_inputs)]
        all_out.extend(stage1_out)
        return all_out

    def forward(self, x):
        if self.first_model.training:
            outputs, aux_outputs = self.first_model(x)
            return outputs, aux_outputs, self.forward_stage2(outputs), self.forward_stage2(aux_outputs)

        else:
            outputs = self.first_model(x)
            return self.forward_stage2(outputs)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes, n_attributes=0, bottleneck=False, expand_dim=0, three_class=False,
                 connect_CY=False):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.n_attributes = n_attributes
        self.bottleneck = bottleneck
        self.expand_dim = expand_dim

        if connect_CY:
            self.cy_fc = FC(n_attributes, num_classes, expand_dim)
        else:
            self.cy_fc = None

        self.all_fc = nn.ModuleList()

        if n_attributes > 0:
            if not bottleneck:  # cotraining
                self.all_fc.append(FC(768, num_classes, expand_dim, stddev=0.001))
            for i in range(self.n_attributes):
                self.all_fc.append(FC(768, 1, expand_dim, stddev=0.001))
        else:
            self.all_fc.append(FC(768, num_classes, expand_dim, stddev=0.001))

    def forward(self, x):
        # N x 768 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 768 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 768
        out = []
        for fc in self.all_fc:
            out.append(fc(x))
        if self.n_attributes > 0 and not self.bottleneck and self.cy_fc is not None:
            attr_preds = torch.cat(out[1:], dim=1)
            out[0] += self.cy_fc(attr_preds)
        return out


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
