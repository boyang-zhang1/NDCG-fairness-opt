import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
import numpy as np
import einops
from typing import Dict    


def simple_fairness_loss(predictions, batch, device, fairness_weight, eps=1e-10,
                         expectation_mode='avg', use_balanced_fairness=True):
    """
    Calculate simple fairness loss based on group attribute differences.

    Args:
        predictions (torch.Tensor): Model predictions with shape [batch_size, slate_length]
        batch (dict): Batch dictionary containing fairness indices
        device (torch.device): Device for tensor operations
        fairness_weight (float): Weight for fairness regularization
        eps (float): Small epsilon to avoid division by zero
        expectation_mode (str): Mode for computing expectation ('avg' or '1')
        use_balanced_fairness (bool): Whether to use balanced fairness constraint

    Returns:
        torch.Tensor: Fairness loss for each batch item [batch_size]
    """
    batch_size, slate_length = predictions.size()
    prediction_softmax = F.softmax(predictions, dim=1)  # [batch_size, slate_length]

    # Get fairness group indices
    group_a_indices = batch['a_index'].to(device)  # [batch_size, slate_length]
    group_b_indices = batch['b_index'].to(device)

    # Calculate group-wise weighted predictions
    group_a_weighted = group_a_indices * prediction_softmax  # [batch_size, slate_length]
    group_b_weighted = group_b_indices * prediction_softmax

    # Initialize expectation weights
    expectation_weight_a = torch.ones(batch_size, device=device)
    expectation_weight_b = torch.ones(batch_size, device=device)

    if expectation_mode == 'avg':
        expectation_weight_a = 1.0 / (torch.sum(group_a_indices, dim=1) + eps)  # [batch_size]
        expectation_weight_b = 1.0 / (torch.sum(group_b_indices, dim=1) + eps)

    # Calculate average exposure for each group
    group_a_exposure = group_a_weighted.sum(dim=1) * expectation_weight_a  # [batch_size]
    group_b_exposure = group_b_weighted.sum(dim=1) * expectation_weight_b

    # Calculate fairness violation
    exposure_difference = group_b_exposure.float() - group_a_exposure.float()  # [batch_size]
    fairness_weight_sqrt = math.sqrt(fairness_weight)

    if use_balanced_fairness:
        # Penalize any difference between groups
        fairness_loss = (exposure_difference * fairness_weight_sqrt) ** 2  # [batch_size]
    else:
        # Only penalize when group B has higher exposure than group A
        fairness_loss = (torch.where(
            exposure_difference > 0, exposure_difference, torch.tensor(0.0, device=device)
        ) * fairness_weight_sqrt) ** 2

    return fairness_loss


class ListNetFairLoss(nn.Module):
    """
    ListNet loss with fairness regularization.
    Combines cross-entropy loss with fairness constraint.
    """

    def __init__(self, num_positive: int, eps: float = 1e-10,
                 fairness_weight: float = 1e5, expectation_mode: str = '1',
                 use_balanced_fairness: bool = True):
        """
        Initialize ListNet Fair Loss.

        Args:
            num_positive (int): Number of positive items
            eps (float): Small epsilon for numerical stability
            fairness_weight (float): Weight for fairness regularization
            expectation_mode (str): Mode for expectation calculation ('1' or 'avg')
            use_balanced_fairness (bool): Whether to use balanced fairness
        """
        super(ListNetFairLoss, self).__init__()
        self.num_positive = num_positive
        self.eps = eps
        self.fairness_weight = fairness_weight
        self.expectation_mode = expectation_mode
        self.use_balanced_fairness = use_balanced_fairness

    def forward(self, predictions: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of ListNet Fair Loss.

        Args:
            predictions (torch.Tensor): Model predictions [batch_size, slate_length]
            batch (Dict[str, torch.Tensor]): Batch dictionary with ratings and fairness info

        Returns:
            torch.Tensor: Combined ListNet and fairness loss
        """
        device = predictions.device
        batch_size, slate_length = predictions.size()

        # Prepare target ratings (positive items have ratings, negatives are zero)
        positive_ratings = batch['rating'][:, :self.num_positive]
        zero_ratings = torch.zeros(batch_size, slate_length - self.num_positive, device=device)
        target_ratings = torch.cat([positive_ratings.float(), zero_ratings], dim=1)

        # Calculate ListNet loss (cross-entropy between rating and prediction distributions)
        prediction_probs = F.softmax(predictions, dim=1)
        rating_probs = F.softmax(target_ratings, dim=1)
        listnet_loss = -torch.sum(rating_probs * torch.log(prediction_probs + self.eps), dim=1)

        # Calculate fairness loss
        fairness_loss = simple_fairness_loss(
            predictions, batch, device, self.fairness_weight,
            eps=self.eps, expectation_mode=self.expectation_mode,
            use_balanced_fairness=self.use_balanced_fairness
        )

        return torch.mean(listnet_loss + fairness_loss)


class BPRFairLoss(nn.Module):
    """
    Bayesian Personalized Ranking (BPR) loss with fairness regularization.
    """

    def __init__(self, eps: float = 1e-10, fairness_weight: float = 1e5,
                 expectation_mode: str = '1', use_balanced_fairness: bool = True):
        """
        Initialize BPR Fair Loss.

        Args:
            eps (float): Small epsilon for numerical stability
            fairness_weight (float): Weight for fairness regularization
            expectation_mode (str): Mode for expectation calculation
            use_balanced_fairness (bool): Whether to use balanced fairness
        """
        super(BPRFairLoss, self).__init__()
        self.eps = eps
        self.fairness_weight = fairness_weight
        self.expectation_mode = expectation_mode
        self.use_balanced_fairness = use_balanced_fairness

    def forward(self, positive_scores: torch.Tensor, negative_scores: torch.Tensor,
                predictions: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of BPR Fair Loss.

        Args:
            positive_scores (torch.Tensor): Scores for positive items
            negative_scores (torch.Tensor): Scores for negative items
            predictions (torch.Tensor): Full prediction tensor for fairness calculation
            batch (Dict[str, torch.Tensor]): Batch dictionary with fairness info

        Returns:
            torch.Tensor: Combined BPR and fairness loss
        """
        device = predictions.device

        # Calculate BPR loss: -log(sigmoid(pos_score - neg_score))
        score_differences = positive_scores[:, None] - negative_scores
        bpr_loss = -score_differences.sigmoid().log().sum(dim=1)

        # Calculate fairness loss
        fairness_loss = simple_fairness_loss(
            predictions, batch, device, self.fairness_weight,
            eps=self.eps, expectation_mode=self.expectation_mode,
            use_balanced_fairness=self.use_balanced_fairness
        )

        return torch.mean(bpr_loss) + torch.mean(fairness_loss)


class ListMLEFairLoss(nn.Module):
    """
    ListMLE (List-wise Maximum Likelihood Estimation) loss with fairness regularization.
    """

    def __init__(self, num_positive: int, eps: float = 1e-10,
                 fairness_weight: float = 1e5, expectation_mode: str = '1',
                 use_balanced_fairness: bool = True):
        """
        Initialize ListMLE Fair Loss.

        Args:
            num_positive (int): Number of positive items
            eps (float): Small epsilon for numerical stability
            fairness_weight (float): Weight for fairness regularization
            expectation_mode (str): Mode for expectation calculation
            use_balanced_fairness (bool): Whether to use balanced fairness
        """
        super(ListMLEFairLoss, self).__init__()
        self.num_positive = num_positive
        self.eps = eps
        self.fairness_weight = fairness_weight
        self.expectation_mode = expectation_mode
        self.use_balanced_fairness = use_balanced_fairness

    def forward(self, predictions: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of ListMLE Fair Loss.

        Args:
            predictions (torch.Tensor): Model predictions [batch_size, slate_length]
            batch (Dict[str, torch.Tensor]): Batch dictionary with ratings and fairness info

        Returns:
            torch.Tensor: Combined ListMLE and fairness loss
        """
        device = predictions.device
        batch_size, slate_length = predictions.size()

        # Prepare target ratings
        positive_ratings = batch['rating'][:, :self.num_positive]
        zero_ratings = torch.zeros(batch_size, slate_length - self.num_positive, device=device)
        target_ratings = torch.cat([positive_ratings.float(), zero_ratings], dim=1)

        # Sort predictions according to rating order (descending)
        sorted_indices = target_ratings.sort(descending=True, dim=1)[1]
        predictions_sorted = torch.gather(predictions, dim=1, index=sorted_indices)

        # Numerical stability: subtract max value
        max_predictions, _ = predictions_sorted.max(dim=1, keepdim=True)
        predictions_normalized = predictions_sorted - max_predictions

        # Calculate cumulative sums for ListMLE loss
        exp_predictions = predictions_normalized.exp()
        cumulative_sums = torch.cumsum(exp_predictions.flip(dims=[1]), dim=1).flip(dims=[1])
        listmle_loss_per_item = torch.log(cumulative_sums + self.eps) - predictions_normalized

        # Calculate fairness loss
        fairness_loss = simple_fairness_loss(
            predictions, batch, device, self.fairness_weight,
            eps=self.eps, expectation_mode=self.expectation_mode,
            use_balanced_fairness=self.use_balanced_fairness
        )

        return torch.mean(torch.sum(listmle_loss_per_item, dim=1) + fairness_loss)


class ListwiseCrossEntropyLoss(nn.Module):
    """
    Listwise Cross-Entropy loss with fairness regularization and momentum updates.
    """

    def __init__(self, num_users: int, num_items: int, num_positive: int,
                 momentum_factor: float, eps: float = 1e-10,
                 fairness_weight: float = 1e5, expectation_mode: str = '1',
                 use_balanced_fairness: bool = True):
        """
        Initialize Listwise Cross-Entropy Loss.

        Args:
            num_users (int): Number of users
            num_items (int): Number of items
            num_positive (int): Number of positive items
            momentum_factor (float): Momentum factor for exponential moving average
            eps (float): Small epsilon for numerical stability
            fairness_weight (float): Weight for fairness regularization
            expectation_mode (str): Mode for expectation calculation
            use_balanced_fairness (bool): Whether to use balanced fairness
        """
        super(ListwiseCrossEntropyLoss, self).__init__()
        self.num_positive = num_positive
        self.momentum_factor = momentum_factor
        self.eps = eps
        self.user_item_statistics = torch.zeros(num_users + 1, num_items + 1)
        self.fairness_weight = fairness_weight
        self.expectation_mode = expectation_mode
        self.use_balanced_fairness = use_balanced_fairness

    def forward(self, predictions: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of Listwise Cross-Entropy Loss.

        Args:
            predictions (torch.Tensor): Model predictions [batch_size, num_pos + num_neg]
            batch (Dict[str, torch.Tensor]): Batch dictionary

        Returns:
            torch.Tensor: Combined cross-entropy and fairness loss
        """
        device = predictions.device
        batch_size = predictions.size(0)

        # Reshape predictions for positive and negative items
        positive_predictions = einops.rearrange(
            predictions[:, :self.num_positive], 'batch pos -> (batch pos) 1'
        )  # [batch_size * num_positive, 1]

        negative_predictions = einops.repeat(
            predictions[:, self.num_positive:], 'batch neg -> (batch pos) neg',
            pos=self.num_positive
        )  # [batch_size * num_positive, num_neg]

        # Calculate margin and exponential margin
        score_margins = negative_predictions - positive_predictions
        exp_margins = torch.exp(score_margins - torch.max(score_margins)).detach_()

        # Get user and item IDs for momentum updates
        user_ids = einops.repeat(
            batch['user_id'], 'batch -> (batch pos)', pos=self.num_positive
        )  # [batch_size * num_positive]

        positive_item_ids = einops.rearrange(
            batch['item_id'][:, :self.num_positive], 'batch pos -> (batch pos)'
        )  # [batch_size * num_positive]

        # Move statistics tensor to device and update with momentum
        self.user_item_statistics = self.user_item_statistics.to(device)
        exp_margin_means = torch.mean(exp_margins, dim=1).to(device)

        # Exponential moving average update
        current_stats = self.user_item_statistics[user_ids, positive_item_ids]
        updated_stats = (
            (1 - self.momentum_factor) * current_stats +
            self.momentum_factor * exp_margin_means
        )
        self.user_item_statistics[user_ids, positive_item_ids] = updated_stats

        # Calculate normalized margins for loss
        normalized_margins = exp_margins / (updated_stats[:, None] + self.eps)

        # Calculate main loss
        main_loss = torch.sum(score_margins * normalized_margins) / batch_size

        # Calculate fairness loss
        fairness_loss = simple_fairness_loss(
            predictions, batch, device, self.fairness_weight,
            eps=self.eps, expectation_mode=self.expectation_mode,
            use_balanced_fairness=self.use_balanced_fairness
        )

        return main_loss + torch.mean(fairness_loss)


class NDCGLoss(nn.Module):
    """
    NDCG (Normalized Discounted Cumulative Gain) Loss with fairness regularization.
    Supports top-k optimization and various fairness constraints.
    """

    def __init__(self, num_users: int, num_items: int, num_positive: int,
                 momentum_gamma: float, gradient_momentum: float = 0.9, learning_rate: float = 0.01,
                 squared_hinge_c: float = 1.0, top_k: int = -1, topk_version: str = 'theo',
                 threshold_tau1: float = 0.001, threshold_tau2: float = 0.0001, eps: float = 1e-10,
                 batch_size: int = 256, psi_function: str = 'softmax', fair_psi_function: str = 'softmax',
                 hinge_margin: float = 2.0, sigmoid_c: float = 2.0, sigmoid_alpha: float = 1.0,
                 sigmoid_beta: float = 1.0, sigmoid_temperature: float = 1.0, expectation_mode: str = '1',
                 fairness_type: str = 'exp_top1_fair', gamma2: float = 0.5, gamma3: float = 0.5,
                 gamma4: float = 0.5, variance_weight: float = 1.0, fairness_weight: float = 1e5,
                 use_balanced_fairness: bool = True, use_simple_fairness: bool = False):
        """
        Initialize NDCG Loss with fairness constraints.

        Args:
            num_users (int): Number of users in the system
            num_items (int): Number of items in the system
            num_positive (int): Number of positive items per training instance
            momentum_gamma (float): Momentum factor for exponential moving average
            gradient_momentum (float): Momentum for gradient estimation
            learning_rate (float): Learning rate for threshold updates
            squared_hinge_c (float): Parameter for squared hinge loss
            top_k (int): Top-k items to consider for NDCG (-1 for all)
            topk_version (str): Version of top-k implementation ('theo' or 'prac')
            threshold_tau1 (float): Temperature parameter for threshold sigmoid
            threshold_tau2 (float): Regularization parameter for threshold
            eps (float): Small epsilon for numerical stability
            batch_size (int): Batch size for training
            psi_function (str): Psi function type ('softmax', 'sigmoid', 'hinge')
            fair_psi_function (str): Fair psi function type
            hinge_margin (float): Margin for hinge loss
            sigmoid_c (float): Sigmoid parameter
            sigmoid_alpha (float): Alpha parameter for sigmoid
            sigmoid_beta (float): Beta parameter for sigmoid
            sigmoid_temperature (float): Temperature for sigmoid
            expectation_mode (str): Mode for expectation calculation
            fairness_type (str): Type of fairness constraint
            gamma2, gamma3, gamma4 (float): Additional momentum parameters
            variance_weight (float): Weight for variance term
            fairness_weight (float): Weight for fairness regularization
            use_balanced_fairness (bool): Whether to use balanced fairness
            use_simple_fairness (bool): Whether to use simple fairness formulation
        """
        super(NDCGLoss, self).__init__()
        # Core parameters
        self.num_positive = num_positive
        self.squared_hinge_c = squared_hinge_c
        self.momentum_gamma = momentum_gamma
        self.top_k = top_k
        self.num_items = num_items
        self.topk_version = topk_version
        self.eps = eps

        # Threshold learning parameters
        self.user_thresholds = torch.zeros(num_users + 1)  # Learnable thresholds for users
        self.threshold_gradients = torch.zeros(num_users + 1)  # Moving average of threshold gradients
        self.threshold_hessians = torch.zeros(num_users + 1)  # Moving average of threshold Hessians
        self.gradient_momentum = gradient_momentum
        self.learning_rate = learning_rate
        self.threshold_tau1 = threshold_tau1
        self.threshold_tau2 = threshold_tau2

        # Sigmoid and activation parameters
        self.psi_function = psi_function
        self.fair_psi_function = fair_psi_function
        self.hinge_margin = hinge_margin
        self.sigmoid_alpha = sigmoid_alpha
        self.sigmoid_beta = sigmoid_beta
        self.sigmoid_temperature = sigmoid_temperature
        self.sigmoid_c = sigmoid_c

        # Fairness parameters
        self.fairness_type = fairness_type
        self.fairness_weight = fairness_weight
        self.expectation_mode = expectation_mode
        self.use_simple_fairness = use_simple_fairness
        self.use_balanced_fairness = use_balanced_fairness

        # Momentum tracking tensors
        self.user_item_statistics = torch.zeros(num_users + 1, num_items + 1)
        self.group_a_statistics = torch.zeros(num_users + 1)
        self.group_b_statistics = torch.zeros(num_users + 1)
        self.denominator_statistics = torch.zeros(num_users + 1)
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.gamma4 = gamma4
        self.variance_weight = variance_weight

        # Additional tensors for specific fairness types
        self.user_item_fairness_statistics = None
        if fairness_type in ['log_rank_fair2', 'log_rank_fair3']:
            self.user_item_fairness_statistics = torch.zeros(num_users + 1, num_items + 1)

        # Debug mode (disabled by default for production)
        self.debug_mode = False

    def _debug_print(self, message: str, value) -> None:
        """
        Print debug information if debug mode is enabled.

        Args:
            message (str): Debug message
            value: Value to print
        """
        if self.debug_mode:
            logging.debug(f"{message}: {value}")

    def _squared_hinge_loss(self, predictions: torch.Tensor, margin: float) -> torch.Tensor:
        """
        Calculate squared hinge loss.

        Args:
            predictions (torch.Tensor): Input predictions
            margin (float): Hinge loss margin

        Returns:
            torch.Tensor: Squared hinge loss values
        """
        return torch.max(torch.zeros_like(predictions), predictions + margin) ** 2

    def _calculate_sigmoid_adjustment(self, group_size: int, total_items: int) -> float:
        """
        Calculate sigmoid adjustment factor for fairness.

        Args:
            group_size (int): Size of the attribute group
            total_items (int): Total number of items

        Returns:
            float: Adjustment factor
        """
        return (total_items - group_size) * 0.5 / total_items

    def forward(self, predictions: torch.Tensor, batch: Dict[str, torch.Tensor], epoch: int) -> torch.Tensor:
        """
        Forward pass of NDCG Loss with fairness constraints.

        Args:
            predictions (torch.Tensor): Model predictions with shape [batch_size, num_positive + num_negative]
            batch (Dict[str, torch.Tensor]): Batch containing user_id, item_id, rating, num_pos_items, ideal_dcg
            epoch (int): Current training epoch

        Returns:
            torch.Tensor: Combined NDCG and fairness loss
        """
        self.debug_mode = False
        device = predictions.device

        # Extract ratings and batch information
        ratings = batch['rating'][:, :self.num_positive]  # [batch_size, num_positive]
        batch_size = ratings.size(0)
        num_positive_items = batch['num_pos_items'].float()  # [batch_size]
        ideal_dcg = batch['ideal_dcg'].float()  # [batch_size]

        # Prepare predictions for ranking loss calculation
        # Expand predictions to compare each positive item against all items
        predictions_expanded = einops.repeat(
            predictions, 'batch items -> (batch pos) items', pos=self.num_positive
        )  # [batch_size * num_positive, num_positive + num_negative]

        positive_predictions = einops.rearrange(
            predictions[:, :self.num_positive], 'batch pos -> (batch pos) 1'
        )  # [batch_size * num_positive, 1]

        # Calculate ranking scores using squared hinge loss
        ranking_scores = torch.mean(
            self._squared_hinge_loss(
                predictions_expanded - positive_predictions, self.squared_hinge_c
            ), dim=-1
        )  # [batch_size * num_positive]

        ranking_scores = ranking_scores.reshape(batch_size, self.num_positive)

        # Calculate DCG gains: G_i = 2^{r_i} - 1
        dcg_gains = (2.0 ** ratings - 1).float()

        # Extract user and item IDs for momentum updates
        user_ids = batch['user_id'].cpu()
        positive_item_ids = batch['item_id'][:, :self.num_positive].cpu()  # [batch_size, num_positive]

        # Flatten for indexing
        positive_item_ids_flat = einops.rearrange(positive_item_ids, 'batch pos -> (batch pos)')
        user_ids_repeated = einops.repeat(user_ids, 'batch -> (batch pos)', pos=self.num_positive)

        # Update user-item statistics with exponential moving average
        self.user_item_statistics[user_ids_repeated, positive_item_ids_flat] = (
            (1 - self.momentum_gamma) * self.user_item_statistics[user_ids_repeated, positive_item_ids_flat] +
            self.momentum_gamma * ranking_scores.clone().detach().reshape(-1).cpu()
        )

        # Get updated statistics and move to device
        updated_statistics = self.user_item_statistics[user_ids_repeated, positive_item_ids_flat].reshape(
            batch_size, self.num_positive
        ).to(device)

        # Calculate NDCG gradient: ∇f(g) where f(g) = 1/log2(1 + |I| * g)
        ndcg_gradient = (dcg_gains * self.num_items) / (
            (torch.log2(1 + self.num_items * updated_statistics)) ** 2 *
            (1 + self.num_items * updated_statistics) * np.log(2)
        )

        ndcg_gradient_backup = ndcg_gradient.clone()
        # Top-k optimization with learnable thresholds
        if self.top_k > 0:
            # Calculate prediction differences from learnable thresholds
            positive_threshold_diffs = (
                predictions[:, :self.num_positive].clone().detach() -
                self.user_thresholds[user_ids][:, None].to(device)
            )
            all_threshold_diffs = (
                predictions.clone().detach() -
                self.user_thresholds[user_ids][:, None].to(device)
            )  # [batch_size, num_positive + num_negative]

            # Calculate threshold gradients using sigmoid activation
            sigmoid_activations = torch.sigmoid(all_threshold_diffs.cpu() / self.threshold_tau1)
            threshold_gradients = (
                self.top_k / self.num_items +
                self.threshold_tau2 * self.user_thresholds[user_ids] -
                torch.mean(sigmoid_activations, dim=-1)
            )

            self._debug_print("Sigmoid values for threshold optimization",
                             (sigmoid_activations.mean().item(), (sigmoid_activations > 0.5).sum().item()))

            # Update threshold statistics with momentum
            self.threshold_gradients[user_ids] = (
                self.gradient_momentum * threshold_gradients +
                (1 - self.gradient_momentum) * self.threshold_gradients[user_ids]
            )

            # Update learnable thresholds
            self.user_thresholds[user_ids] = (
                self.user_thresholds[user_ids] -
                self.learning_rate * self.threshold_gradients[user_ids]
            )

            # Apply psi function for practical top-k version
            if self.topk_version == 'prac':
                if self.psi_function == 'hinge':
                    psi_weights = torch.max(
                        positive_threshold_diffs + self.hinge_margin,
                        torch.zeros_like(positive_threshold_diffs)
                    )
                elif self.psi_function == 'sigmoid':
                    psi_weights = self.sigmoid_alpha * torch.sigmoid(
                        positive_threshold_diffs * self.sigmoid_alpha
                    )
                else:
                    raise ValueError(f"Psi function '{self.psi_function}' is not supported")

                ndcg_gradient *= psi_weights

            # Apply psi function for theoretical top-k version with second-order optimization
            elif self.topk_version == 'theo':
                if self.psi_function == 'hinge':
                    psi_weights = torch.max(
                        positive_threshold_diffs + self.hinge_margin,
                        torch.zeros_like(positive_threshold_diffs)
                    )
                    psi_derivatives = (positive_threshold_diffs + self.hinge_margin > 0).float()

                elif self.psi_function == 'sigmoid':
                    sigmoid_values = torch.sigmoid(positive_threshold_diffs * self.sigmoid_alpha)
                    psi_weights = self.sigmoid_alpha * sigmoid_values
                    psi_derivatives = self.sigmoid_alpha * sigmoid_values * (1 - sigmoid_values)

                    # Calculate temperature term for Hessian approximation
                    sigmoid_all = torch.sigmoid(all_threshold_diffs / self.threshold_tau1)
                    temp_term = sigmoid_all * (1 - sigmoid_all) / self.threshold_tau1

                    self._debug_print("Positive predictions sigmoid",
                                     (sigmoid_values.mean().item(), (sigmoid_values > 0.5).sum().item()))

                elif self.psi_function == 'softmax':
                    psi_weights = F.softmax(positive_threshold_diffs * self.sigmoid_alpha, dim=-1)
                    psi_derivatives = self.sigmoid_alpha * psi_weights
                    temp_term = F.softmax(all_threshold_diffs / self.threshold_tau1, dim=-1) / self.threshold_tau1

                else:
                    raise ValueError(f"Psi function '{self.psi_function}' is not supported")

                # Apply psi weighting to NDCG gradient
                ndcg_gradient *= psi_weights

                # Calculate Hessian approximation for second-order optimization
                self._debug_print('Temperature term size', temp_term.size())
                hessian_approximation = self.threshold_tau2 + torch.mean(temp_term, dim=1)

                # Update Hessian statistics
                self.threshold_hessians[user_ids] = (
                    self.gradient_momentum * hessian_approximation.cpu() +
                    (1 - self.gradient_momentum) * self.threshold_hessians[user_ids]
                )

                # Calculate second-order correction term
                hessian_correction = (
                    torch.mean(temp_term * predictions, dim=1) /
                    self.threshold_hessians[user_ids].to(device)
                )

                # Calculate DCG values for second-order term
                dcg_values = -dcg_gains / torch.log2(1 + self.num_items * updated_statistics)

                # Compute main loss with second-order correction
                main_loss_terms = (
                    ndcg_gradient * ranking_scores +
                    psi_derivatives * dcg_values * (
                        predictions[:, :self.num_positive] - hessian_correction[:, None]
                    )
                )

                main_loss = (
                    num_positive_items * torch.mean(main_loss_terms, dim=-1) / ideal_dcg
                ).mean()




                # Calculate fairness loss based on fairness type
                if self.use_simple_fairness:
                    fairness_loss = torch.mean(
                        simple_fairness_loss(
                            predictions, batch, device, self.fairness_weight, eps=self.eps,
                            expectation_mode=self.expectation_mode,
                            use_balanced_fairness=self.use_balanced_fairness
                        )
                    )
                else:
                    if self.fair_type == 'exp_topk':
                        a_index = batch['a_index'].to(device)
                        b_index = batch['b_index'].to(device)
                        mask_ratio_a = batch['mask_ratio_a'].to(device)
                        mask_ratio_b = batch['mask_ratio_b'].to(device)
                        rho = batch['rho'].to(device)
                        exy_a = torch.full([batch_size], 1)
                        exy_b = torch.full([batch_size], 1)
                        if self.e_mode == 'avg':
                            exy_a = 1 / (torch.sum(a_index, dim=1) + self.eps)  # [batch_size]
                            exy_b = 1 / (torch.sum(b_index, dim=1) + self.eps)
                        if self.fair_psi_func == 'softmax':
                            qa = torch.mean(F.softmax(preds_lambda_diffs * self.sigmoid_beta,  dim=-1)
                                            * a_index, dim=-1) * exy_a / mask_ratio_a
                            qb = torch.mean(F.softmax(preds_lambda_diffs * self.sigmoid_beta, dim=-1)
                                            * b_index, dim=-1) * exy_b / mask_ratio_b

                            weight_2 = self.sigmoid_beta * F.softmax(preds_lambda_diffs * self.sigmoid_beta, dim=-1)
                        else:
                            temp = torch.sigmoid(preds_lambda_diffs * self.sigmoid_beta)
                            qa = torch.mean(temp * a_index, dim=-1) * exy_a / mask_ratio_a
                            qb = torch.mean(temp * b_index, dim=-1) * exy_b / mask_ratio_b
                            self._debug_print("Fairness sigmoid values",
                                             (temp.mean().item(), (temp > 0.5).sum().item()))

                            weight_2 = self.sigmoid_beta * torch.sigmoid(preds_lambda_diffs * self.sigmoid_beta) * (
                                    1 - torch.sigmoid(preds_lambda_diffs * self.sigmoid_beta))
                        pa = torch.mean(predictions * weight_2 * a_index, dim=-1) * exy_a / mask_ratio_a
                        pb = torch.mean(predictions * weight_2 * b_index, dim=-1) * exy_b / mask_ratio_b

                        pa_hat = torch.mean(weight_2 * a_index, dim=-1) * exy_a / mask_ratio_a
                        pb_hat = torch.mean(weight_2 * b_index, dim=-1) * exy_b / mask_ratio_b

                        root_c = math.sqrt(self.fairness_c)
                        loss_user = (root_c * (rho * qa - qb)) * (
                                root_c * (rho * pa - pb + rho * hessian_term * pa_hat - hessian_term * pb_hat))

                        loss_g2 = torch.mean(loss_user, dim=-1)

                    elif self.fair_type == 'exp_top1_fair':
                        a_index = batch['a_index'].to(device)
                        b_index = batch['b_index'].to(device)

                        batch_size, slate_length = predictions.size()
                        pred_exp = torch.exp(predictions / self.sigmoid_t) # [256*305]

                        a_map = a_index * pred_exp
                        b_map = b_index * pred_exp
                        a_avg = a_map.sum(dim=1) / (torch.sum(a_index, dim=1) + self.eps)
                        b_avg = b_map.sum(dim=1) / (torch.sum(b_index, dim=1) + self.eps)
                        
                        d_avg = torch.mean(pred_exp, dim=1)

                        self.u_a[user_ids] = (1 - self.gamma2) * self.u_a[user_ids] + self.gamma2 * a_avg.cpu()
                        self.u_b[user_ids] = (1 - self.gamma3) * self.u_b[user_ids] + self.gamma3 * b_avg.cpu()
                        self.u_d[user_ids] = (1 - self.gamma4) * self.u_d[user_ids] + self.gamma4 * d_avg.cpu()

                        u_a = self.u_a[user_ids].to(device)
                        u_b = self.u_b[user_ids].to(device)
                        u_d = self.u_d[user_ids].to(device)

                        grad_f3_constant = self.fairness_c * (u_a - u_b) * self.v1 / u_d / slate_length 
                        grad_f3_1 = (grad_f3_constant * self.v1 / slate_length / u_d).detach_()
                        grad_f3_2 = (-grad_f3_constant * self.v1 / slate_length / u_d).detach_()
                        grad_f3_3 = (grad_f3_constant * (-u_a + u_b) * self.v1 / slate_length / (u_d ** 2)).detach_()

                        loss_g2 = torch.mean(a_avg * grad_f3_1 + b_avg * grad_f3_2 + d_avg * grad_f3_3, dim=-1)

                    elif self.fair_type == 'exp_top1_fair_topk':
                        # If preds_lambda_diffs > 0, then temp = 1, else 0
                        # temp1 = (preds_lambda_diffs > 0).float()
                        temp_sigmoid = torch.sigmoid(preds_lambda_diffs * self.sigmoid_beta)
                        temp1 = temp_sigmoid

                        a_index = batch['a_index'].to(device)
                        b_index = batch['b_index'].to(device)

                        batch_size, slate_length = predictions.size()
                        pred_exp = torch.exp(predictions)  # [256*305]

                        a_map_sigmoid = a_index * pred_exp * temp_sigmoid
                        b_map_sigmoid = b_index * pred_exp * temp_sigmoid
                        a_avg_sigmoid = a_map_sigmoid.sum(dim=1) / (torch.sum(a_index, dim=1) + self.eps)
                        b_avg_sigmoid = b_map_sigmoid.sum(dim=1) / (torch.sum(b_index, dim=1) + self.eps)
                        d_avg = torch.mean(pred_exp, dim=1)

                        #for top K revise amap bmap
                        with torch.no_grad():
                            a_map = a_index * pred_exp * temp1
                            b_map = b_index * pred_exp * temp1
                            a_avg = a_map.sum(dim=1) / (torch.sum(a_index, dim=1) + self.eps)
                            b_avg = b_map.sum(dim=1) / (torch.sum(b_index, dim=1) + self.eps)

                            self.u_a[user_ids] = (1 - self.gamma2) * self.u_a[user_ids] + self.gamma2 * a_avg.cpu()
                            self.u_b[user_ids] = (1 - self.gamma3) * self.u_b[user_ids] + self.gamma3 * b_avg.cpu()
                            self.u_d[user_ids] = (1 - self.gamma4) * self.u_d[user_ids] + self.gamma4 * d_avg.cpu()

                            u_a = self.u_a[user_ids].to(device)
                            u_b = self.u_b[user_ids].to(device)
                            u_d = self.u_d[user_ids].to(device)
                            grad_f3_constant = self.fairness_c * (u_a - u_b) * self.v1 / u_d / slate_length 
                            grad_f3_1 = (grad_f3_constant * self.v1 / slate_length / u_d).clone().detach_()
                            grad_f3_2 = (-grad_f3_constant * self.v1 / slate_length / u_d).clone().detach_()
                            grad_f3_3 = (grad_f3_constant * (-u_a + u_b) * self.v1 / slate_length / (u_d ** 2)).clone().detach_()

                        loss_g2 = torch.mean(a_avg_sigmoid * grad_f3_1 + b_avg_sigmoid * grad_f3_2 + d_avg * grad_f3_3, dim=-1)

                    elif self.fair_type == 'rank_fair':
                        a_index = batch['a_index'].to(device)
                        b_index = batch['b_index'].to(device)

                        batch_size, slate_length = predictions.size()

                        predictions_expand_fair = einops.repeat(predictions, 'b n -> (b copy) n',
                                           copy=slate_length)  # [batch_size*slate_length, slate_length]
                        predictions_fair = einops.rearrange(predictions, 'b n -> (b n) 1')  # [batch_size*slate_length, 1]
                        # print("predictions_expand_fair[0]", predictions_expand_fair[0])
                        # print("predictions_fair[0]", predictions_fair[0])

                        g_rank = torch.mean(self._squared_hinge_loss(predictions_expand_fair - predictions_fair, self.sqh_c),
                                    dim=-1)  # [batch_size*slate_length]
                        g_rank = g_rank.reshape(batch_size, slate_length)  # [batch_size, slate_length], line 5 in Algo 2.
                        # print("g_rank[0]", g_rank[0])

                        a_g_map = a_index * g_rank # [batch_size, slate_length]
                        b_g_map = b_index * g_rank
                        a_g_avg = a_g_map.sum(dim=1) / (torch.sum(a_index, dim=1) + self.eps) / slate_length # [batch_size]
                        b_g_avg = b_g_map.sum(dim=1) / (torch.sum(b_index, dim=1) + self.eps) / slate_length

                        self.u_a[user_ids] = (1 - self.gamma2) * self.u_a[user_ids] + self.gamma2 * a_g_avg.cpu()
                        self.u_b[user_ids] = (1 - self.gamma3) * self.u_b[user_ids] + self.gamma3 * b_g_avg.cpu()

                        u_a = self.u_a[user_ids].to(device)
                        u_b = self.u_b[user_ids].to(device)

                        grad_f3_constant = self.fairness_c * (u_a - u_b) * slate_length 
                        grad_f3_1 = grad_f3_constant.detach_()
                        grad_f3_2 = -grad_f3_constant.detach_()

                        loss_g2 = torch.mean(a_g_avg * grad_f3_1 + b_g_avg * grad_f3_2, dim=-1)
                        # print("loss_g2", loss_g2)


                    elif self.fair_type == 'rank_fair_topk':
                        temp_sigmoid = torch.sigmoid(preds_lambda_diffs * self.sigmoid_beta)
                        a_index = batch['a_index'].to(device)
                        b_index = batch['b_index'].to(device)

                        batch_size, slate_length = predictions.size()

                        predictions_expand_fair = einops.repeat(predictions, 'b n -> (b copy) n',
                                           copy=slate_length)  # [batch_size*slate_length, slate_length]
                        predictions_fair = einops.rearrange(predictions[:, :], 'b n -> (b n) 1')  # [batch_size*slate_length, 1]

                        g_rank = torch.mean(self._squared_hinge_loss(predictions_expand_fair - predictions_fair, self.sqh_c),
                                    dim=-1)  # [batch_size*slate_length]
                        g_rank = g_rank.reshape(batch_size, slate_length)  # [batch_size, slate_length], line 5 in Algo 2.

                        a_g_map = a_index * g_rank * temp_sigmoid # [batch_size, num_pos + num_neg]
                        b_g_map = b_index * g_rank * temp_sigmoid
                        a_g_avg = a_g_map.sum(dim=1) / (torch.sum(a_index, dim=1) + self.eps) / slate_length # [batch_size]
                        b_g_avg = b_g_map.sum(dim=1) / (torch.sum(b_index, dim=1) + self.eps) / slate_length                                             

                        self.u_a[user_ids] = (1 - self.gamma2) * self.u_a[user_ids] + self.gamma2 * a_g_avg.cpu()
                        self.u_b[user_ids] = (1 - self.gamma3) * self.u_b[user_ids] + self.gamma3 * b_g_avg.cpu()

                        u_a = self.u_a[user_ids].to(device)
                        u_b = self.u_b[user_ids].to(device)

                        grad_f3_constant = self.fairness_c * (u_a - u_b) * slate_length 
                        grad_f3_1 = grad_f3_constant.detach_()
                        grad_f3_2 = -grad_f3_constant.detach_()

                        loss_g2 = torch.mean(a_g_avg * grad_f3_1 + b_g_avg * grad_f3_2, dim=-1)

                    elif self.fair_type == 'log_rank_fair1':
                        a_index = batch['a_index'].to(device)
                        b_index = batch['b_index'].to(device)

                        batch_size, slate_length = predictions.size()

                        predictions_expand_fair = einops.repeat(predictions, 'b n -> (b copy) n',
                                           copy=slate_length)  # [batch_size*slate_length, slate_length]
                        predictions_fair = einops.rearrange(predictions, 'b n -> (b n) 1')  # [batch_size*slate_length, 1]
                        # print("predictions_expand_fair[0]", predictions_expand_fair[0])
                        # print("predictions_fair[0]", predictions_fair[0])

                        g_rank = 1 / (torch.log2(1 + torch.mean(self._squared_hinge_loss(predictions_expand_fair - predictions_fair, self.sqh_c), dim=-1)))  # [batch_size*slate_length]
                        g_rank = g_rank.reshape(batch_size, slate_length)  # [batch_size, slate_length], line 5 in Algo 2.
                        # print("g_rank[0]", g_rank[0])

                        a_g_map = a_index * g_rank # [batch_size, slate_length]
                        b_g_map = b_index * g_rank
                        a_g_avg = a_g_map.sum(dim=1) / (torch.sum(a_index, dim=1) + self.eps) / slate_length # [batch_size]
                        b_g_avg = b_g_map.sum(dim=1) / (torch.sum(b_index, dim=1) + self.eps) / slate_length

                        self.u_a[user_ids] = (1 - self.gamma2) * self.u_a[user_ids] + self.gamma2 * a_g_avg.cpu()
                        self.u_b[user_ids] = (1 - self.gamma3) * self.u_b[user_ids] + self.gamma3 * b_g_avg.cpu()

                        u_a = self.u_a[user_ids].to(device)
                        u_b = self.u_b[user_ids].to(device)

                        grad_f3_constant = self.fairness_c * (u_a - u_b) * slate_length 
                        grad_f3_1 = grad_f3_constant.detach_()
                        grad_f3_2 = -grad_f3_constant.detach_()

                        loss_g2 = torch.mean(a_g_avg * grad_f3_1 + b_g_avg * grad_f3_2, dim=-1)
                        # print("loss_g2", loss_g2)
                    
                    elif self.fair_type == 'log_rank_fair2':
                        a_index = batch['a_index'].to(device)
                        b_index = batch['b_index'].to(device)

                        batch_size, slate_length = predictions.size()

                        predictions_expand_fair = einops.repeat(predictions, 'b n -> (b copy) n',
                                           copy=slate_length)  # [batch_size*slate_length, slate_length]
                        predictions_flat = einops.rearrange(predictions, 'b n -> (b n) 1')  # [batch_size*slate_length, 1]
                        # print("predictions_expand_fair[0]", predictions_expand_fair[0])
                        # print("predictions_fair[0]", predictions_fair[0])

                        g_rank = torch.mean(self._squared_hinge_loss(predictions_expand_fair - predictions_flat, self.sqh_c), dim=-1)  # [batch_size*slate_length]
                        g_rank = g_rank.reshape(batch_size, slate_length)  # [batch_size, slate_length], line 5 in Algo 2.
                        # print("g_rank[0]", g_rank[0])

                        # user_ids = batch['user_id'].cpu()
                        user_ids_repeat = einops.repeat(user_ids, 'n -> (n copy)', copy=slate_length)

                        item_ids = batch['item_id'].cpu()  # [batch_size, slate_length]
                        item_ids = einops.rearrange(item_ids, 'b n -> (b n)')

                        self.ui[user_ids_repeat, item_ids] = (1 - self.gamma0) * self.ui[
                            user_ids_repeat, item_ids] + self.gamma0 * g_rank.clone().detach_().reshape(-1).cpu()
                        
                        g_ui = self.ui[user_ids_repeat, item_ids].reshape(batch_size, slate_length).to(device)

                        nabla_f_g_fair = self.item_num / ((torch.log2(1 + self.item_num * g_ui)) ** 2 * (1 + self.item_num * g_ui) * np.log(2)) 

                        a_index_num = torch.sum(a_index, dim=-1) + self.eps
                        b_index_num = torch.sum(b_index, dim=-1) + self.eps

                        temp_u1 = (a_index / torch.log2(1 + self.item_num * g_ui)).sum(dim=-1) / a_index_num
                        temp_u2 = (b_index / torch.log2(1 + self.item_num * g_ui)).sum(dim=-1) / b_index_num
                        temp_u1_u2 = temp_u1 - temp_u2

                        loss_user = (a_index * nabla_f_g_fair * g_rank).sum(dim=-1) / a_index_num - (b_index * nabla_f_g_fair * g_rank).sum(dim=-1) / b_index_num

                        loss_g2 = torch.mean(self.fairness_c * temp_u1_u2 * loss_user, dim=-1)

                    elif self.fair_type == 'log_rank_fair3':
                        a_index = batch['a_index'].to(device)
                        b_index = batch['b_index'].to(device)

                        batch_size, slate_length = predictions.size()

                        predictions_expand_fair = einops.repeat(predictions, 'b n -> (b copy) n',
                                           copy=slate_length)  # [batch_size*slate_length, slate_length]
                        predictions_flat = einops.rearrange(predictions, 'b n -> (b n) 1')  # [batch_size*slate_length, 1]
                        # print("predictions_expand_fair[0]", predictions_expand_fair[0])
                        # print("predictions_fair[0]", predictions_fair[0])

                        g_rank = torch.mean(self._squared_hinge_loss(predictions_expand_fair - predictions_flat, self.sqh_c), dim=-1)  # [batch_size*slate_length]
                        g_rank = g_rank.reshape(batch_size, slate_length)  # [batch_size, slate_length], line 5 in Algo 2.
                        # print("g_rank[0]", g_rank[0])

                        # user_ids = batch['user_id'].cpu()
                        user_ids_repeat = einops.repeat(user_ids, 'n -> (n copy)', copy=slate_length)

                        item_ids = batch['item_id'].cpu()  # [batch_size, slate_length]
                        item_ids = einops.rearrange(item_ids, 'b n -> (b n)')

                        self.ui[user_ids_repeat, item_ids] = (1 - self.gamma0) * self.ui[
                            user_ids_repeat, item_ids] + self.gamma0 * g_rank.clone().detach_().reshape(-1).cpu()
                        
                        g_ui = self.ui[user_ids_repeat, item_ids].reshape(batch_size, slate_length).to(device)

                        
                        # a_g_map = a_index * g_rank # [batch_size, slate_length]
                        # b_g_map = b_index * g_rank

                        # a_g_map_log = a_index * 1/ torch.log2(1 + g_ui.clone.detach_()) # [batch_size, slate_length]
                        a_g_map_log = a_index * 1/ torch.log2(1 + g_ui)
                        b_g_map_log = b_index * 1/ torch.log2(1 + g_ui)


                        a_g_avg_log = a_g_map_log.sum(dim=1) / (torch.sum(a_index, dim=1) + self.eps) / slate_length # [batch_size]
                        b_g_avg_log = b_g_map_log.sum(dim=1) / (torch.sum(b_index, dim=1) + self.eps) / slate_length

                        #self.u_a[user_ids] = torch.log2(1 + g_rank.clone().detach_().reshape(-1).cpu()), dim=-1) *amap

                        self.u_a[user_ids] = (1 - self.gamma2) * self.u_a[user_ids] + self.gamma2 * a_g_avg_log.cpu()
                        self.u_b[user_ids] = (1 - self.gamma3) * self.u_b[user_ids] + self.gamma3 * b_g_avg_log.cpu()
                        temp_u_a = self.u_a[user_ids].to(device)
                        temp_u_b = self.u_b[user_ids].to(device)

                        nabla_f_g_fair = -self.item_num / ((torch.log2(1 + self.item_num * g_ui)) ** 2 * (1 + self.item_num * g_ui) * np.log(2)) 

                        a_index_num = torch.sum(a_index, dim=-1) + self.eps
                        b_index_num = torch.sum(b_index, dim=-1) + self.eps

                        temp_u1_u2 = temp_u_a - temp_u_b

                        loss_user = (a_index * nabla_f_g_fair * g_rank).sum(dim=-1) / a_index_num - (b_index * nabla_f_g_fair * g_rank).sum(dim=-1) / b_index_num

                        loss_g2 = torch.mean(self.fairness_c * temp_u1_u2 * loss_user, dim=-1)
                    

                self._debug_print("Main loss and fairness loss", (main_loss.item(), fairness_loss.item()))
                return main_loss + fairness_loss

        # Simple case without top-k constraints
        simple_loss = (
            num_positive_items * torch.mean(ndcg_gradient * ranking_scores, dim=-1) / ideal_dcg
        ).mean()
        return simple_loss
