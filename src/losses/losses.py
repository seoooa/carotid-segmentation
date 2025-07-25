from monai.losses import DiceLoss, DiceFocalLoss, DiceCELoss, SoftDiceclDiceLoss
import torch
import torch.nn as nn
from typing import Callable

# Import skeleton recall components from the original implementation
import sys
import os

from src.losses.skeleton_recall.nnunetv2.training.loss.compound_losses import DC_SkelREC_and_CE_loss
from src.losses.skeleton_recall.nnunetv2.training.loss.dice import SoftSkeletonRecallLoss, MemoryEfficientSoftDiceLoss
from src.losses.skeleton_recall.nnunetv2.utilities.helpers import softmax_helper_dim1

# Import custom skeletonize module
from src.losses.cldice.skeletonize import Skeletonize

class DiceSkeletonRecallLoss(nn.Module):
    """
    Wrapper for the original DC_SkelREC_and_CE_loss to make it compatible with MONAI
    """
    def __init__(self, weight_ce=1, weight_dice=1, weight_srec=1, ignore_label=None):
        super(DiceSkeletonRecallLoss, self).__init__()
        
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
            raise ValueError("target_skel is required for DiceSkeletonRecallLoss")
        
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


def soft_dice(y_true: torch.Tensor, y_pred: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Function to compute soft dice loss
    Compatible with MONAI's SoftDiceclDiceLoss implementation
    """
    intersection = torch.sum((y_true * y_pred)[:, 1:, ...])
    coeff = (2.0 * intersection + smooth) / (torch.sum(y_true[:, 1:, ...]) + torch.sum(y_pred[:, 1:, ...]) + smooth)
    soft_dice: torch.Tensor = 1.0 - coeff
    return soft_dice


class CustomSoftDiceclDiceLoss(SoftDiceclDiceLoss):
    """
    Custom implementation of SoftDiceclDiceLoss that uses the Skeletonize class
    from skeletonize.py instead of the default soft_skel function.
    
    Inherits from MONAI's SoftDiceclDiceLoss and overrides the forward method
    to use custom skeletonization.
    """
    
    def __init__(self, iter_=3, alpha=0.5, smooth=1.0, 
                 probabilistic=True, beta=0.33, tau=1.0, 
                 simple_point_detection='Boolean', num_iter=5):
        """
        Args:
            iter_: Number of iterations for skeletonization (for compatibility, maps to num_iter)
            alpha: Weighing factor for cldice
            smooth: Smoothing parameter
            probabilistic: Whether the input should be binarized using reparametrization trick
            beta: Scale of added logistic noise during reparametrization trick
            tau: Boltzmann temperature for reparametrization trick
            simple_point_detection: Method for simple point detection ('Boolean' or 'EulerCharacteristic')
            num_iter: Number of iterations for the Skeletonize class (if None, uses iter_)
        """
        super().__init__(iter_=iter_, alpha=alpha, smooth=smooth)
        
        # Initialize the custom skeletonize module
        if num_iter is None:
            num_iter = iter_
            
        self.skeletonize = Skeletonize(
            probabilistic=probabilistic,
            beta=beta,
            tau=tau,
            simple_point_detection=simple_point_detection,
            num_iter=num_iter
        )
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Custom forward pass using Skeletonize class instead of soft_skel
        Args:
            y_pred: Model predictions (logits) - shape: [batch, channels, H, W, D]
            y_true: Ground truth labels - shape: [batch, H, W, D] or [batch, 1, H, W, D]
        """
        # Preprocess inputs to ensure they are in [0, 1] range
        # Convert y_pred to probabilities if needed (apply softmax)
        if y_pred.requires_grad:  # If it's logits, apply softmax
            y_pred_prob = torch.softmax(y_pred, dim=1)
        else:
            y_pred_prob = y_pred
            
        # Convert y_true to one-hot if needed
        # Check if y_true contains only integer values (even if dtype is float)
        y_true_unique = torch.unique(y_true)
        is_integer_labels = torch.all(y_true_unique == y_true_unique.round())
        
        if is_integer_labels and len(y_true_unique) <= y_pred.shape[1]:
            # Convert integer labels to one-hot
            num_classes = y_pred.shape[1]
            y_true_onehot = torch.zeros_like(y_pred)
            
            # Ensure y_true has the same number of dims as y_pred
            if y_true.ndim == y_pred.ndim - 1:  # Add channel dimension if needed
                y_true = y_true.unsqueeze(1)
            elif y_true.shape[1] == 1 and y_pred.shape[1] > 1:
                # y_true has 1 channel, need to convert to integer indices
                y_true = y_true.squeeze(1).long()  # Remove channel dim and convert to long
                y_true_onehot = torch.zeros(y_pred.shape, device=y_pred.device, dtype=y_pred.dtype)
                y_true_onehot.scatter_(1, y_true.unsqueeze(1), 1)
            else:
                y_true_onehot.scatter_(1, y_true.long(), 1)
        else:
            # Assume y_true is already in one-hot or probability format
            y_true_onehot = y_true
            
        # Ensure values are in [0, 1] range
        y_pred_prob = torch.clamp(y_pred_prob, 0.0, 1.0)
        y_true_onehot = torch.clamp(y_true_onehot, 0.0, 1.0)
        
        # Compute soft dice loss
        dice = soft_dice(y_true_onehot, y_pred_prob, self.smooth)
        
        # Apply skeletonization channel-wise (skip background channel 0)
        skel_pred_list = []
        skel_true_list = []
        
        for c in range(1, y_pred_prob.shape[1]):  # Skip background channel (index 0)
            # Extract single channel and add channel dimension back for skeletonize
            pred_c = y_pred_prob[:, c:c+1, ...]  # [batch, 1, H, W, D]
            true_c = y_true_onehot[:, c:c+1, ...]  # [batch, 1, H, W, D]
            
            # Apply skeletonization to single channel
            try:
                if pred_c.numel() > 0 and true_c.numel() > 0:
                    skel_pred_c = self.skeletonize(pred_c)
                    skel_true_c = self.skeletonize(true_c)
                    
                    skel_pred_list.append(skel_pred_c)
                    skel_true_list.append(skel_true_c)
                else:
                    # Use original channel if tensors are empty
                    skel_pred_list.append(pred_c)
                    skel_true_list.append(true_c)
            except Exception as e:
                # Use original channel if skeletonization fails
                skel_pred_list.append(pred_c)
                skel_true_list.append(true_c)
        
        # Concatenate skeletonized channels (add background channel of zeros)
        if skel_pred_list:
            background_shape = (y_pred_prob.shape[0], 1) + y_pred_prob.shape[2:]
            skel_pred_bg = torch.zeros(background_shape, device=y_pred_prob.device, dtype=y_pred_prob.dtype)
            skel_true_bg = torch.zeros(background_shape, device=y_pred_prob.device, dtype=y_pred_prob.dtype)
            
            skel_pred = torch.cat([skel_pred_bg] + skel_pred_list, dim=1)
            skel_true = torch.cat([skel_true_bg] + skel_true_list, dim=1)
        else:
            # Fallback: use original tensors if no channels were processed
            skel_pred = y_pred_prob
            skel_true = y_true_onehot
        
        # Compute precision and sensitivity for clDice
        tprec = (torch.sum(torch.multiply(skel_pred, y_true_onehot)[:, 1:, ...]) + self.smooth) / (
            torch.sum(skel_pred[:, 1:, ...]) + self.smooth
        )
        tsens = (torch.sum(torch.multiply(skel_true, y_pred_prob)[:, 1:, ...]) + self.smooth) / (
            torch.sum(skel_true[:, 1:, ...]) + self.smooth
        )
        
        # Compute clDice
        cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)
        
        # Combine dice and clDice
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
        elif loss_name == "SkeletonRecallLoss":
            return SoftSkeletonRecallLossWrapper(
                apply_nonlin=softmax_helper_dim1,
                batch_dice=False,
                do_bg=False,
                smooth=1e-5,
                ddp=False
            )
        elif loss_name == "DiceSkeletonRecallLoss":
            return DiceSkeletonRecallLoss(
                weight_ce=1,
                weight_dice=1,
                weight_srec=1,
                ignore_label=None
            )
        elif loss_name == "SoftDiceclDiceLoss":
            return CustomSoftDiceclDiceLoss(
                iter_=3,
                alpha=0.5,
                smooth=1.0,
                probabilistic=True,
                beta=0.33,
                tau=1.0,
                simple_point_detection='Boolean',
                num_iter=3
            )
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")