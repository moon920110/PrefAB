import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_prob = inputs.gather(-1, targets.unsqueeze(1)).squeeze(1)  # target에 해당하는 class에 대해 예측된 log_prob 추출
        prob = torch.exp(log_prob)
        loss = -self.alpha * (1 - prob) ** self.gamma * log_prob

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class OrdinalCrossEntropyLoss(nn.Module):
    def __init__(self, cutpoints, reduction='mean'):
        super(OrdinalCrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.cutpoints = cutpoints

    def forward(self, y_pred, y_true):
        y_true_0 = y_true == 0
        y_pred_0 = torch.sigmoid(self.cutpoints[0] - y_pred)

        y_true_1 = y_true == 1
        y_pred_1 = torch.sigmoid(self.cutpoints[1] - y_pred) - torch.sigmoid(self.cutpoints[0] - y_pred)

        y_true_2 = y_true == 2
        y_pred_2 = 1 - torch.sigmoid(self.cutpoints[1] - y_pred)

        if self.reduction == 'sum':
            loss_0 = -torch.log(y_pred_0[y_true_0]).sum()
            loss_1 = -torch.log(y_pred_1[y_true_1]).sum()
            loss_2 = -torch.log(y_pred_2[y_true_2]).sum()
        elif self.reduction == 'mean':
            loss_0 = -torch.log(y_pred_0[y_true_0]).mean()
            loss_1 = -torch.log(y_pred_1[y_true_1]).mean()
            loss_2 = -torch.log(y_pred_2[y_true_2]).mean()
        else:
            loss_0 = -torch.log(y_pred_0[y_true_0])
            loss_1 = -torch.log(y_pred_1[y_true_1])
            loss_2 = -torch.log(y_pred_2[y_true_2])

        return (loss_0 + loss_1 + loss_2)/3
