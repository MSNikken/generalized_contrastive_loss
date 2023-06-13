import torch.nn as nn
import torch
import torch.nn.functional as F


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class BaseNet(nn.Module):
    def __init__(self, backbone, global_pool=None, poolkernel=7, norm=None, p=3):
        super(BaseNet, self).__init__()
        self.backbone = backbone
        for name, param in self.backbone.named_parameters():
            n = param.size()[0]
        self.feature_length = n
        if global_pool == "max":
            self.pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        elif global_pool == "avg":
            self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        elif global_pool == "GeM":
            self.pool = GeM(p=p)
        else:
            self.pool = None
        self.norm = norm

    def forward(self, x0):
        out = self.backbone.forward(x0)
        out = self.pool.forward(out).squeeze(-1).squeeze(-1)
        if self.norm == "L2":
            out = nn.functional.normalize(out)
        return out


class SiameseNet(BaseNet):
    def __init__(self, backbone, global_pool=None, poolkernel=7,norm=None, p=3):
        super(SiameseNet, self).__init__(backbone, global_pool, poolkernel, norm=norm, p=p)
    
    def forward_single(self, x0):
        return super(SiameseNet, self).forward(x0)

    def forward(self, x0, x1):
        out0 = super(SiameseNet, self).forward(x0)
        out1 = super(SiameseNet, self).forward(x1)
        return out0, out1


class Predictor(nn.Module):
    def __init__(self, in_features=2048, out_features=4):
        super(Predictor, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.softmax(x)
        return self.activation(x)


class Projector(nn.Module):
    def __init__(self, in_features=2048, out_features=1024):
        super(Projector, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.BatchNorm1d(out_features)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        return self.activation(x)


class SiamesePredictiveNet(BaseNet):
    def __init__(self, backbone, global_pool=None, poolkernel=7, norm=None, p=3):
        super(SiamesePredictiveNet, self).__init__(backbone, global_pool, poolkernel, norm=norm, p=p)
        self.predictor = Predictor()
        self.projector = Projector()

    def forward_single(self, x0):
        return super(SiamesePredictiveNet, self).forward(x0)

    def forward(self, x_c0, x_c1, x_p):
        out_c0 = self.projector(super(SiamesePredictiveNet, self).forward(x_c0))
        out_c1 = self.projector(super(SiamesePredictiveNet, self).forward(x_c1))
        out_p = self.predictor(super(SiamesePredictiveNet, self).forward(x_p))
        return out_c0, out_c1, out_p

