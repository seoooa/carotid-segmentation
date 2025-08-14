from monai.losses import DiceLoss, DiceFocalLoss, DiceCELoss
from src.losses.cldice import soft_skel
from monai.networks.utils import one_hot
import torch
import warnings

class ModifiedSoftDiceclDiceLoss(torch.nn.modules.loss._Loss):
    def __init__(self, to_onehot_y=False, softmax=False, include_background=True, 
                 iter_=3, alpha=0.5, smooth=1e-5):
        super().__init__()
        self.to_onehot_y = to_onehot_y
        self.softmax = softmax
        self.include_background = include_background
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.to_onehot_y:
            n_pred_ch = input.shape[1]
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)
        
        if self.softmax:
            input = torch.softmax(input, dim=1)
            
        # Dice Loss 계산
        start_channel = 0 if self.include_background else 1
        intersection = torch.sum((target * input)[:, start_channel:, ...])
        denominator = torch.sum(target[:, start_channel:, ...]) + torch.sum(input[:, start_channel:, ...])
        dice = 1.0 - (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        
        # CL-Dice Loss 계산
        skel_pred = soft_skel(input, self.iter)
        skel_true = soft_skel(target, self.iter)
        
        tprec = (torch.sum(torch.multiply(skel_pred, target)[:, start_channel:, ...]) + self.smooth) / (
            torch.sum(skel_pred[:, start_channel:, ...]) + self.smooth
        )
        tsens = (torch.sum(torch.multiply(skel_true, input)[:, start_channel:, ...]) + self.smooth) / (
            torch.sum(skel_true[:, start_channel:, ...]) + self.smooth
        )
        cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)
        
        # 최종 Loss 계산
        total_loss: torch.Tensor = (1.0 - self.alpha) * dice + self.alpha * cl_dice
        return total_loss

class LossFactory:
    @staticmethod
    def create_loss(loss_name):
        if loss_name == "DiceLoss":
            return DiceLoss(to_onehot_y=True, softmax=True)
        elif loss_name == "DiceCELoss":
            return DiceCELoss(to_onehot_y=True, softmax=True)
        elif loss_name == "DiceFocalLoss":
            return DiceFocalLoss(to_onehot_y=True, softmax=True)
        elif loss_name == "SoftDiceclDiceLoss":
            return ModifiedSoftDiceclDiceLoss(
                to_onehot_y=True,
                softmax=True,
                include_background=True,
                iter_=3,
                alpha=0.5,
                smooth=1e-5
            )
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")