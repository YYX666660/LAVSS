import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()

    def forward(self, preds, targets, weight=None):
        if isinstance(preds, list):
            N = len(preds)
            if weight is None:
                weight = preds[0].new_ones(1)

            errs = [self._forward(preds[n], targets[n], weight)
                    for n in range(N)]
            err = torch.mean(torch.stack(errs))

        elif isinstance(preds, torch.Tensor):
            if weight is None:
                weight = preds.new_ones(1)
            err = self._forward(preds, targets, weight)

        return err

class L1Loss(BaseLoss):
    def __init__(self):
        super(L1Loss, self).__init__()

    def _forward(self, pred, target, weight):
        return torch.mean(weight * torch.abs(pred - target))

class L2Loss(BaseLoss):
    def __init__(self):
        super(L2Loss, self).__init__()

    def _forward(self, pred, target, weight):
        return torch.mean(weight * torch.pow(pred - target, 2))

class BCELoss(BaseLoss):
    def __init__(self):
        super(BCELoss, self).__init__()

    def _forward(self, pred, target, weight):
        return F.binary_cross_entropy(pred, target, weight=weight)

class BCEWithLogitsLoss(BaseLoss):
    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()

    def _forward(self, pred, target, weight):
        return F.binary_cross_entropy_with_logits(pred, target, weight=weight)

class CELoss(BaseLoss):
    def __init__(self):
        super(CELoss, self).__init__()

    def _forward(self, pred, target, weight=None):
        return F.cross_entropy(pred, target)

def SISNR_loss(pred, label):
    sisnr_loss = 0.
    B, C, _ = label.shape
    loss_list = torch.zeros(B)
    label = label.to(pred.device)
    for b in range(pred.shape[0]):
        num_speaker_2 = torch.all(label[b, 0] == label[b, 1])
        loss_00 = cal_SISNR(pred[b, 0], label[b, 0])
        loss_10 = cal_SISNR(pred[b, 1], label[b, 0])
        if not num_speaker_2:
            loss_11 = cal_SISNR(pred[b, 1], label[b, 1])
            loss_01 = cal_SISNR(pred[b, 0], label[b, 1])
            if ((loss_00 + loss_11) <= (loss_10 + loss_01)):
                loss_list[b] = (loss_00 + loss_11) / 2
                # print(f'inside the loss function, the SISNR loss for this audio is: {loss_00} and {loss_11}')
            else:
                loss_list[b] = (loss_10 + loss_01) / 2
                # print(f'inside the loss function, the SISNR loss for this audio is: {loss_10} and {loss_01}')
        else:
            if loss_00 <= loss_10:
                loss_list[b] = loss_00
                # print(f'inside the loss function, the SISNR loss for this audio is: {loss_00}')
            else:
                loss_list[b] = loss_10
                # print(f'inside the loss function, the SISNR loss for this audio is: {loss_10}')
        sisnr_loss = torch.mean(loss_list)
    return sisnr_loss

def cal_SISNR(preds, target):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        preds: numpy.ndarray, [B, C, T]
        target: numpy.ndarray, [B, C, T]
    Returns:
        SISNR
    """
    assert preds.shape == target.shape
    EPS = 1e-10

    target = target - torch.mean(target, dim=-1, keepdim=True)
    preds = preds - torch.mean(preds, dim=-1, keepdim=True)

    alpha = (torch.sum(preds * target, dim=-1, keepdim=True) + EPS) / (
        torch.sum(target ** 2, dim=-1, keepdim=True) + EPS)
    target_scaled = alpha * target

    noise = target_scaled - preds

    val = (torch.sum(noise ** 2, dim=-1) + EPS) / (torch.sum(target_scaled ** 2, dim=-1) + EPS)
    val = 10 * torch.log10(val)
    loss = val
    return loss


