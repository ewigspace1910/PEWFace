#head
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output

############ArcFace##############
class ArcFace(nn.Module):
    """Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight) #init weight
        self.eps = 1e-7
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m) #threshold
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_t = F.linear(F.normalize(input), F.normalize(self.weight, dim=1))
        cos_t = cos_t.clamp(-1. + self.eps, 1. - self.eps)
        sin_t = torch.sqrt(1.0 - torch.pow(cos_t, 2))
        cos_phi = cos_t * self.cos_m - sin_t * self.sin_m
        cos_phi = torch.where(cos_t > self.th, cos_phi, cos_t - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros_like(cos_phi)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = torch.where(one_hot == 1, cos_phi, cos_t)
        output *= self.s

        return output


###########CosFace##############
class CosFace(nn.Module):
    """Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=64.0, m=0.35):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.eps = 1e-7

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_t = F.linear(F.normalize(input), F.normalize(self.weight, dim=1)).clamp(-1, 1)  # for numerical stability
        cos_phi = cos_t - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = F.one_hot(label, num_classes=self.out_features)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = torch.where(one_hot==1, cos_phi, cos_t)
        output *= self.s
        # print(output)

        return output


###########MagFace##############
class MagFace(nn.Module):
    """ implement Magface https://arxiv.org/pdf/2103.06627.pdf"""

    def __init__(self, in_features, out_features, s=64.0, l_a=10, u_a=110, l_m=0.45, u_m=0.8, lambda_g=35):
        super(MagFace, self).__init__()

        self.l_a = l_a
        self.u_a = u_a
        self.l_m = l_m
        self.u_m = u_m
        
        self.lambda_g = lambda_g
        self.s = s
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.eps = 1e-7

    def compute_m(self, a):
        return (self.u_m - self.l_m) / (self.u_a - self.l_a) * (a - self.l_a) + self.l_m

    def compute_g(self, a):
        return torch.mean( (1 / self.u_a**2) * a + 1 / a)

    def forward(self, input, label):
        
        cos_t = F.linear(F.normalize(input), F.normalize(self.weight, dim=1)).clamp(-1. + self.eps, 1. - self.eps)
        sin_t = torch.sqrt(1.0 - torch.pow(cos_t, 2))

        # compute additive margin
        a = torch.norm(input, dim=1, keepdim=True).clamp(self.l_a, self.u_a)
        m = self.compute_m(a)
        cos_m, sin_m = torch.cos(m), torch.sin(m)

        g = self.compute_g(a)

        #threshold when phi > 180 
        threshold = torch.cos(math.pi - cos_m)
        mm = torch.sin(math.pi - m) * m

        # phi = theta + m(a) => cos(phi)
        cos_phi = cos_t * cos_m - sin_t * sin_m
        cos_phi = torch.where(cos_phi > threshold, cos_phi, cos_t - mm)

        # one-hot label
        one_hot = torch.zeros_like(cos_phi)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        # build the output logits
        output = one_hot * cos_phi + (1.0 - one_hot) * cos_t
        # feature re-scaling
        output *= self.s

        return output, self.lambda_g * g

###########ElasTicFACE##########
class ElasticArcFace(nn.Module):
    """ implement ElasticFace  https://github.com/fdbtrs/ElasticFace """
    def __init__(self, in_features, out_features, s=64.0, m=0.50,std=0.0125,plus=False):
        super(ElasticArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.normal_(self.weight, std=0.01)
        self.std=std
        self.plus=plus
        self.eps = 1e-7

    def forward(self, input, label):
        cos_t = F.linear(F.normalize(input), F.normalize(self.weight, dim=1)).clamp(-1. + self.eps, 1. - self.eps)
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_t.size()[1], device=cos_t.device)
        # margin = torch.normal(mean=self.m, std=self.std, size=label[index, None].size(), device=cos_t.device) 
        margin = torch.normal(mean=self.m, std=self.std, size=label[index, None].size(), device=cos_t.device).clamp(self.m-self.std, self.m+self.std) # Fast converge .clamp(self.m-self.std, self.m+self.std)

        if self.plus:
            with torch.no_grad():
                distmat = cos_t[index, label.view(-1)].detach().clone()
                _, idicate_cosie = torch.sort(distmat, dim=0, descending=True)
                margin, _ = torch.sort(margin, dim=0)
            m_hot.scatter_(1, label[index, None], margin[idicate_cosie])
        else:
            m_hot.scatter_(1, label[index, None], margin)
        
        cos_t.acos_()
        cos_t[index] += m_hot

        return cos_t.cos_().mul_(self.s)


class ElasticCosFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.35,std=0.0125, plus=False):
        super(ElasticCosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.std=std
        self.plus=plus
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.normal_(self.weight, std=0.01)
        self.eps = 1e-7

    def forward(self, input, label):
        cos_t = F.linear(F.normalize(input), F.normalize(self.weight, dim=1)).clamp(-1. + self.eps, 1. - self.eps) 
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_t.size()[1], device=cos_t.device)
        #margin = torch.normal(mean=self.m, std=self.std, size=label[index, None].size(), device=cos_t.device) 
        margin = torch.normal(mean=self.m, std=self.std, size=label[index, None].size(), device=cos_t.device).clamp(self.m-self.std, self.m+self.std) # Fast converge 
        
        if self.plus:
            with torch.no_grad():
                distmat = cos_t[index, label.view(-1)].detach().clone()
                _, idicate_cosie = torch.sort(distmat, dim=0, descending=True)
                margin, _ = torch.sort(margin, dim=0)
            m_hot.scatter_(1, label[index, None], margin[idicate_cosie])
        else:
            m_hot.scatter_(1, label[index, None], margin)
        
        cos_t[index] -= m_hot
        return cos_t * self.s

############AdaFace##############
class AdaFace(nn.Module):
    def __init__(self, in_features, out_features, m=0.4, h=0.333, s=64., t_alpha=1.0):
        super(AdaFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m 
        self.eps = 1e-7
        self.h = h
        self.weight = Parameter(torch.FloatTensor(out_features ,in_features))
        nn.init.normal_(self.weight, std=0.01)
        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_std', torch.ones(1)*100)

    def forward(self, input, label):
        cos_t = F.linear(F.normalize(input), F.normalize(self.weight, dim=1)).clamp(-1. + self.eps, 1. - self.eps) 
        safe_norms = torch.clip(torch.norm(input, dim=1, keepdim=True), min=0.001, max=100).clone().detach()

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        #Image Quality Indicator
        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)

        # g_angular
        m_arc = torch.zeros(label.size()[0], cos_t.size()[1], device=cos_t.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = cos_t.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cos_t = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(label.size()[0], cos_t.size()[1], device=cos_t.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cos_t = cos_t - m_cos
        return cos_t * self.s

############other##############
class AdaCos(nn.Module):
    def __init__(self, in_features, out_features, m=0.50):
        super(AdaCos, self).__init__()
        self.num_features = in_features
        self.n_classes = out_features
        self.s = math.sqrt(2) * math.log(out_features - 1)
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.eps = 1e-6

    def forward(self, input, label=None):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_t = F.linear(F.normalize(input), F.normalize(self.weight, dim=1)).clamp(-1. + self.eps, 1. - self.eps)

        
        theta = torch.acos(cos_t)
        one_hot = torch.zeros_like(cos_t)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # rescale logits - auto modification
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * cos_t), torch.zeros_like(cos_t))
            B_avg = torch.sum(B_avg) / input.size(0)
            # print(B_avg)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
            #print(self.s)
        
        cos_phi = cos_t - self.m
        output = torch.where(one_hot==1, cos_phi, cos_t)
        output = self.s * cos_t
        return output