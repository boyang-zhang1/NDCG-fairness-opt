import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from typing import Dict


class Listwise_CE_Loss(nn.Module):
    def __init__(self, user_num: int, item_num: int, num_pos: int, gamma0: float, eps: float=1e-10):
        super(Listwise_CE_Loss, self).__init__()
        self.num_pos = num_pos
        self.gamma0 = gamma0
        self.eps = eps
        self.u = torch.zeros(user_num+1, item_num+1)

    def forward(self, predictions: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = predictions.size(0)
        pos_pred = einops.rearrange(predictions[:, :self.num_pos], 'b n -> (b n) 1')                    # [batch_size*num_pos, 1]
        neg_pred = einops.repeat(predictions[:, self.num_pos:], 'b n -> (b copy) n', copy=self.num_pos) # [batch_size*num_pos, num_neg]
        margin = neg_pred - pos_pred
        exp_margin = torch.exp(margin - torch.max(margin)).detach_()
        
        user_ids = einops.repeat(batch['user_id'], 'b -> (b copy)', copy=self.num_pos).cpu()         # [batch_size*num_pos]
        pos_item_ids = einops.rearrange(batch['item_id'][:, :self.num_pos], 'b n -> (b n)').cpu()    # [batch_size*num_pos]

        self.u[user_ids, pos_item_ids] = (1-self.gamma0) * self.u[user_ids, pos_item_ids] + self.gamma0 * torch.mean(exp_margin, dim=1).cpu()

        exp_margin_softmax = exp_margin / (self.u[user_ids, pos_item_ids][:, None].cuda() + self.eps)

        loss = torch.sum(margin * exp_margin_softmax)
        loss /= batch_size

        return loss


class NDCG_Loss(nn.Module):
    def __init__(self, user_num: int, item_num: int, num_pos: int,
                 gamma0: float, gamma1: float=0.9, eta0: float=0.01,
                 sqh_c: float=1.0, k: int=-1, topk_version: str='theo', tau_1: float=0.001, tau_2: float=0.0001,
                 psi_func: str='sigmoid', hinge_margin: float=2.0, c_sigmoid: float=2.0, sigmoid_alpha: float=2.0):
        super(NDCG_Loss, self).__init__()
        self.num_pos = num_pos
        self.sqh_c = sqh_c
        self.gamma0 = gamma0
        self.k = k                               
        self.lambda_q = torch.zeros(user_num+1)  # learnable thresholds for all querys (users)
        self.v_q = torch.zeros(user_num+1)       # moving average estimator for \nabla_{\lambda} L_q
        self.gamma1 = gamma1                        
        self.tau_1 = tau_1                            
        self.tau_2 = tau_2                       
        self.eta0 = eta0                  
        self.item_num = item_num
        self.topk_version = topk_version         # theo: sigmoid_alpha=2.0 ; prac: sigmoid_alpha=0.01
        self.s_q = torch.zeros(user_num+1)       # moving average estimator for \nabla_{\lambda}^2 L_q
        self.psi_func = psi_func
        self.hinge_margin = hinge_margin
        self.sigmoid_alpha = sigmoid_alpha
        self.c_sigmoid = c_sigmoid
        self.u = torch.zeros(user_num+1, item_num+1)

    def _squared_hinge_loss(self, x, c):
        return torch.max(torch.zeros_like(x), x + c) ** 2

    def forward(self, predictions: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            predictions:  predicted socres from the model, shape: [batch_size, num_pos + num_neg]
            batch:        a dict that contains the following keys: user_id, item_id, rating, num_pos_items, ideal_dcg        
        """
        device = predictions.device
        ratings = batch['rating'][:, :self.num_pos]                                                           # [batch_size, num_pos]
        batch_size = ratings.size()[0]
        predictions_expand = einops.repeat(predictions, 'b n -> (b copy) n', copy=self.num_pos)  # [batch_size*num_pos, num_pos+num_neg]
        predictions_pos = einops.rearrange(predictions[:, :self.num_pos], 'b n -> (b n) 1')      # [batch_suze*num_pos, 1]

        num_pos_items = batch['num_pos_items'].float()  # [batch_size], the number of positive items for each user
        ideal_dcg = batch['ideal_dcg'].float()          # [batch_size], the ideal dcg for each user
        
        g = torch.mean(self._squared_hinge_loss(predictions_expand-predictions_pos, self.sqh_c), dim=-1)   # [batch_size*num_pos]
        g = g.reshape(batch_size, self.num_pos)                                                            # [batch_size, num_pos], line 5 in Algo 2.

        G = (2.0 ** ratings - 1).float()

        user_ids = batch['user_id']
        user_ids = user_ids.cpu()
        pos_item_ids = batch['item_id'][:, :self.num_pos]  # [batch_size, num_pos]
        pos_item_ids = pos_item_ids.cpu()

        pos_item_ids = einops.rearrange(pos_item_ids, 'b n -> (b n)')
        user_ids_repeat = einops.repeat(user_ids, 'n -> (n copy)', copy=self.num_pos)

        self.u[user_ids_repeat, pos_item_ids] = (1-self.gamma0) * self.u[user_ids_repeat, pos_item_ids] + self.gamma0 * g.clone().detach_().reshape(-1).cpu()
        g_u = self.u[user_ids_repeat, pos_item_ids].reshape(batch_size, self.num_pos).to(device)

        nabla_f_g = (G * self.item_num) / ((torch.log2(1 + self.item_num*g_u))**2 * (1 + self.item_num*g_u) * np.log(2)) # \nabla f(g)

        if self.k > 0:
            pos_preds_lambda_diffs = predictions[:, :self.num_pos].clone().detach_() - self.lambda_q[user_ids][:, None].to(device)
            preds_lambda_diffs = predictions.clone().detach_() - self.lambda_q[user_ids][:, None].to(device)

            # the gradient of lambda
            grad_lambda_q = self.k/self.item_num + self.tau_2*self.lambda_q[user_ids] - torch.mean(torch.sigmoid(preds_lambda_diffs.cpu() / self.tau_1), dim=-1)
            self.v_q[user_ids] = self.gamma1 * grad_lambda_q + (1-self.gamma1) * self.v_q[user_ids]
            self.lambda_q[user_ids] = self.lambda_q[user_ids] - self.eta0 * self.v_q[user_ids]

            if self.topk_version == 'prac':
                if self.psi_func == 'hinge':
                    nabla_f_g *= torch.max(pos_preds_lambda_diffs+self.hinge_margin, torch.zeros_like(pos_preds_lambda_diffs))
                elif self.psi_func == 'sigmoid':
                    nabla_f_g *= self.c_sigmoid * torch.sigmoid(pos_preds_lambda_diffs * self.sigmoid_alpha)
                else:
                    assert 0, "psi_func " + self.psi_func + " is not supported."

            elif self.topk_version == 'theo':
                if self.psi_func == 'hinge':
                    nabla_f_g *= torch.max(pos_preds_lambda_diffs+self.hinge_margin, torch.zeros_like(pos_preds_lambda_diffs))
                    weight_1 = (pos_preds_lambda_diffs+self.hinge_margin > 0).float()
                elif self.psi_func == 'sigmoid':
                    nabla_f_g *= self.c_sigmoid * torch.sigmoid(pos_preds_lambda_diffs * self.sigmoid_alpha)
                    weight_1 = self.c_sigmoid * torch.sigmoid(pos_preds_lambda_diffs * self.sigmoid_alpha) * (1 - torch.sigmoid(pos_preds_lambda_diffs * self.sigmoid_alpha))
                else:
                    assert 0, "psi_func " + self.psi_func + " is not supported."

                temp_term = torch.sigmoid(preds_lambda_diffs / self.tau_1) * (1 - torch.sigmoid(preds_lambda_diffs / self.tau_1)) / self.tau_1
                L_lambda_hessian = self.tau_2 + torch.mean(temp_term, dim=1)
                self.s_q[user_ids] = self.gamma1 * L_lambda_hessian.cpu() + (1-self.gamma1) * self.s_q[user_ids]
                hessian_term = torch.mean(temp_term * predictions, dim=1) / self.s_q[user_ids].to(device)
                f_g_u = -G / torch.log2(1 + self.item_num*g_u)
                loss = (num_pos_items * torch.mean(nabla_f_g * g + weight_1 * f_g_u * (predictions[:, :self.num_pos] - hessian_term[:, None]), dim=-1) / ideal_dcg).mean()
                return loss

        loss = (num_pos_items * torch.mean(nabla_f_g * g, dim=-1) / ideal_dcg).mean()
        return loss
