import torch
import torch.nn.functional as F
from torch import nn
import math

class WeightedWaveletLoss(nn.Module):
    """Weighted loss for wavelet noise predictions"""
    def __init__(self, ll_weight=2.0, lh_weight=1.0, hl_weight=1.0, hh_weight=0.5, 
                 loss_type='l1'):
        super().__init__()
        self.ll_weight = ll_weight
        self.lh_weight = lh_weight  
        self.hl_weight = hl_weight
        self.hh_weight = hh_weight
        self.loss_type = loss_type
        
    def forward(self, pred, target):
        # Split into wavelet subbands (16 channels from 4-channel images)
        pred_ll = pred[:, 0:4, :, :]    # Low-low
        pred_lh = pred[:, 4:8, :, :]    # Low-high
        pred_hl = pred[:, 8:12, :, :]   # High-low
        pred_hh = pred[:, 12:16, :, :] # High-high
        
        target_ll = target[:, 0:4, :, :]
        target_lh = target[:, 4:8, :, :]
        target_hl = target[:, 8:12, :, :]
        target_hh = target[:, 12:16, :, :]
        
        # Compute weighted losses
        if self.loss_type == 'l1':
            loss_ll = F.l1_loss(pred_ll, target_ll, reduction='sum') * self.ll_weight
            loss_lh = F.l1_loss(pred_lh, target_lh, reduction='sum') * self.lh_weight
            loss_hl = F.l1_loss(pred_hl, target_hl, reduction='sum') * self.hl_weight
            loss_hh = F.l1_loss(pred_hh, target_hh, reduction='sum') * self.hh_weight
        else:  # l2/mse
            loss_ll = F.mse_loss(pred_ll, target_ll, reduction='sum') * self.ll_weight
            loss_lh = F.mse_loss(pred_lh, target_lh, reduction='sum') * self.lh_weight
            loss_hl = F.mse_loss(pred_hl, target_hl, reduction='sum') * self.hl_weight
            loss_hh = F.mse_loss(pred_hh, target_hh, reduction='sum') * self.hh_weight

        loss = loss_ll + loss_lh + loss_hl + loss_hh
        weights = self.ll_weight + self.lh_weight + self.hl_weight + self.hh_weight
      
        if not torch.isfinite(loss):
            print("Skipping step due to invalid loss.")
            loss = torch.zeros((), device=loss.device, requires_grad=True)
        else:
            loss = loss / weights

        return loss