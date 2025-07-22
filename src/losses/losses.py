from monai.losses import DiceLoss, DiceFocalLoss, DiceCELoss
import torch
import torch.nn as nn
from typing import Callable

# Import skeleton recall components from the original implementation
import sys
import os

from src.losses.skeleton_recall.nnunetv2.training.loss.compound_losses import DC_SkelREC_and_CE_loss
from src.losses.skeleton_recall.nnunetv2.training.loss.dice import SoftSkeletonRecallLoss, MemoryEfficientSoftDiceLoss
from src.losses.skeleton_recall.nnunetv2.utilities.helpers import softmax_helper_dim1


class SkeletonRecallLossWrapper(nn.Module):
    """
    Wrapper for the original DC_SkelREC_and_CE_loss to make it compatible with MONAI
    """
    def __init__(self, weight_ce=1, weight_dice=1, weight_srec=1, ignore_label=None):
        super(SkeletonRecallLossWrapper, self).__init__()
        
        soft_dice_kwargs = {'batch_dice': False, 'smooth': 1e-5, 'do_bg': False, 'ddp': False}
        soft_skelrec_kwargs = {'batch_dice': False, 'smooth': 1e-5, 'do_bg': False, 'ddp': False}
        ce_kwargs = {}
        
        self.loss = DC_SkelREC_and_CE_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            soft_skelrec_kwargs=soft_skelrec_kwargs,
            ce_kwargs=ce_kwargs,
            weight_ce=weight_ce,
            weight_dice=weight_dice,
            weight_srec=weight_srec,
            ignore_label=ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss
        )
    
    def forward(self, input, target, target_skel=None):
        """
        input: network prediction (b, c, h, w, d)
        target: segmentation ground truth (b, h, w, d) or (b, 1, h, w, d)
        target_skel: skeleton ground truth (b, h, w, d) or (b, 1, h, w, d) - required
        """
        if target_skel is None:
            raise ValueError("target_skel is required for SkeletonRecallLossWrapper")
        
        # Ensure target has channel dimension
        if target.ndim == input.ndim - 1:
            target = target.unsqueeze(1)
        if target_skel.ndim == input.ndim - 1:
            target_skel = target_skel.unsqueeze(1)
        
        return self.loss(input, target, target_skel)


class SoftSkeletonRecallLossWrapper(nn.Module):
    """
    Wrapper for the original SoftSkeletonRecallLoss to make it compatible with MONAI
    """
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=False, smooth=1e-5, ddp=False):
        super(SoftSkeletonRecallLossWrapper, self).__init__()
        
        # Use the original SoftSkeletonRecallLoss from skeleton-recall GitHub
        if apply_nonlin is None:
            apply_nonlin = softmax_helper_dim1
            
        self.loss = SoftSkeletonRecallLoss(
            apply_nonlin=apply_nonlin,
            batch_dice=batch_dice,
            do_bg=do_bg,
            smooth=smooth,
            ddp=ddp
        )
    
    def forward(self, input, target_skel, loss_mask=None):
        """
        input: network prediction (b, c, h, w, d)
        target_skel: skeleton ground truth (b, h, w, d) or (b, 1, h, w, d)
        loss_mask: optional mask
        """
        # Ensure target_skel has channel dimension
        if target_skel.ndim == input.ndim - 1:
            target_skel = target_skel.unsqueeze(1)
        
        return self.loss(input, target_skel, loss_mask=loss_mask)


class LossFactory:
    @staticmethod
    def create_loss(loss_name):
        if loss_name == "DiceLoss":
            return DiceLoss(to_onehot_y=True, softmax=True)
        elif loss_name == "DiceCELoss":
            return DiceCELoss(to_onehot_y=True, softmax=True)
        elif loss_name == "DiceFocalLoss":
            return DiceFocalLoss(to_onehot_y=True, softmax=True)
        elif loss_name == "SkeletonRecallLoss":
            return SoftSkeletonRecallLossWrapper(
                apply_nonlin=softmax_helper_dim1,
                batch_dice=False,
                do_bg=False,
                smooth=1e-5,
                ddp=False
            )
        elif loss_name == "DC_SkelREC_and_CE_loss":
            return SkeletonRecallLossWrapper(
                weight_ce=1,
                weight_dice=1,
                weight_srec=1,
                ignore_label=None
            )
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")