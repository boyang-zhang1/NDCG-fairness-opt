# -*- coding: UTF-8 -*-

# Standard library imports
import gc
import json
import logging
import math
import multiprocessing as mp
import os
from multiprocessing import Pool
from time import time
from typing import Dict, List, NoReturn, Tuple

# Third-party imports
import numpy as np
from numpy import ndarray
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
from models.BaseModel import BaseModel
from utils import utils

# Constants
TOLERANCE_EPSILON = 1e-5

def sample_ranking(probability_matrix):
    """
    Generate a stochastic ranking by sampling from a probability matrix.

    This function implements a position-wise sampling strategy where for each position
    in the ranking, items are sampled according to their probabilities, and selected
    items are removed from subsequent sampling rounds to ensure no duplicates.

    Args:
        probability_matrix (np.ndarray): 2D probability matrix of shape (n_positions, n_items)
                                       where entry [i,j] represents the probability of
                                       placing item j at position i

    Returns:
        list: Sampled ranking indices representing the order of items
    """
    ranking = []
    remaining_indices = list(range(probability_matrix.shape[0]))

    for i in range(probability_matrix.shape[0]):
        current_probs = probability_matrix[i, remaining_indices]
        if current_probs.sum() != 0:
            current_probs /= current_probs.sum()
        next_index = np.random.choice(len(remaining_indices), p=current_probs)
        ranking.append(remaining_indices.pop(next_index))

    return ranking


class BaseRunner(object):
    """
    Base runner class for training and evaluating recommendation models with fairness considerations.

    This class provides comprehensive functionality for:
    - Training recommendation models with various optimization strategies
    - Evaluating models using multiple ranking metrics (NDCG, MAP)
    - Computing fairness metrics across sensitive groups
    - Managing training processes with early stopping and model checkpointing
    - Supporting various fairness-aware evaluation modes

    The runner supports both accuracy-focused and fairness-aware model training,
    with configurable evaluation strategies and baseline comparisons.
    """

    @staticmethod
    def parse_runner_args(parser):
        """
        Parse command-line arguments for the recommendation system runner.

        Configures training hyperparameters, evaluation settings, fairness metrics,
        and system optimization parameters for the recommendation model.

        Args:
            parser: ArgumentParser instance to add arguments to

        Returns:
            ArgumentParser: Updated parser with all runner-specific arguments
        """
        parser.add_argument('--epoch', type=int, default=120,
                            help='Number of epochs.')
        parser.add_argument('--check_epoch', type=int, default=1,
                            help='Check some tensors every check_epoch.')
        parser.add_argument('--test_epoch', type=int, default=-1,
                            help='Print test results every test_epoch (-1 means no print).')
        parser.add_argument('--early_stop', type=int, default=-1,
                            help='The number of epochs when dev results drop continuously.')
        parser.add_argument('--early_stop_thresh', type=int, default=20,
                            help='The number of epochs when dev leave from the best.')
        parser.add_argument('--lr', type=float, default=0.0004,
                            help='Learning rate.')
        parser.add_argument('--schedule', default='[60]', type=str,
                            help='Learning rate schedule (when to drop lr by 4x)')
        parser.add_argument('--l2', type=float, default=0,
                            help='Weight decay in optimizer.')
        parser.add_argument('--batch_size', type=int, default=256,
                            help='Batch size during training.')
        parser.add_argument('--eval_batch_size', type=int, default=256,
                            help='Batch size during testing.')
        parser.add_argument('--optimizer', type=str, default='Adam',
                            help='optimizer: SGD, Adam, Adagrad, Adadelta')
        parser.add_argument('--clip_value', type=float, default=0.0,
                            help='value for gradient clipping')
        parser.add_argument('--num_workers', type=int, default=8,
                            help='Number of processors when prepare batches in DataLoader')
        parser.add_argument('--pin_memory', type=int, default=1,
                            help='pin_memory in DataLoader')
        parser.add_argument('--topk', type=str, default='5,20,50,200,305',
                            help='The number of items recommended to each user.')
        parser.add_argument('--metric', type=str, default='NDCG,MAP',
                            help='metrics: NDCG, MAP')
        parser.add_argument('--fair_result_mode', type=str, default='1',
                            help='mode for fairness result, default is 1 for abs(a-b), '
                                 '"avg" for average, "exp_norm" for exp normalize')
        parser.add_argument('--fairness_metric', type=str, default='exp_norm',
                            help='metric for fairness evaluation, default is "exp_norm" for exp/sum(exp) normalize, '
                                 '"count" for count diff, "sigmoid" for sigmoid, "ndcg_diff" for ndcg_diff, '
                                 '"sigmoid_thresh" for sigmoid with the top num_pos threshold')
        parser.add_argument('--prefetch_factor', type=int, default=2, help='prefetch factor for dataloader.')
        parser.add_argument('--model_modes', type=str, default='fn,f',
                            help='f=only fairness, n=only ndcg, fn=1/(1-fair)+1/ndcg.')
        parser.add_argument('--baseline', type=int, default=-1,
                            help='print baseline test result, available: 0: color blind')
        parser.add_argument('--data_p', type=float, default=0.28,
                            help='the dataset sensitive group proportion')
        parser.add_argument('--threshold_k', type=int, default=5,
                            help='The number of threshold_k items during fair evaluation.')
        parser.add_argument('--fairness_loss_type', type=str, default='mae',
                            help='Type of fairness loss calculation: "mae" for Mean Absolute Error, "mse" for Mean Squared Error')
        return parser

    def _MAP_at_k(self, hit: np.ndarray, ground_truth_rank: np.ndarray) -> float:
        """
        Calculate Mean Average Precision at k (MAP@k).

        MAP@k measures the quality of ranked retrieval by computing the average
        precision across all relevant items within the top-k positions. It considers
        both the ranking order and the number of relevant items retrieved.

        Args:
            hit (np.ndarray): Binary hit matrix of shape (n_users, k) indicating
                            whether each top-k item is relevant (1) or not (0)
            ground_truth_rank (np.ndarray): Ranking positions of ground truth items
                                          of shape (n_users, n_relevant_items)

        Returns:
            float: Mean Average Precision at k across all users
        """
        ap_list = []
        hit_ground_truth_rank = (hit * ground_truth_rank).astype(float)
        sorted_hit_ground_truth_rank = np.sort(hit_ground_truth_rank)
        for idx, row in enumerate(sorted_hit_ground_truth_rank):
            precision_list = []
            counter = 1
            for item in row:
                if item > 0:
                    precision_list.append(counter / item)
                    counter += 1
            ap = np.sum(precision_list) / np.sum(hit[idx]) if np.sum(hit[idx]) > 0 else 0
            ap_list.append(ap)
        return np.mean(ap_list)

    @staticmethod
    def _NDCG_at_k(ratings: np.ndarray, normalizer_mat: np.ndarray, hit: np.ndarray, ground_truth_rank: np.ndarray,
                   k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at k (NDCG@k).

        NDCG@k is a ranking metric that measures the quality of a ranking by considering
        both the relevance (rating) of items and their positions. Higher-rated items
        at higher positions contribute more to the score, with logarithmic discounting
        applied to lower positions.

        Args:
            ratings (np.ndarray): User-item rating matrix of shape (n_users, n_items)
                                containing relevance scores for each user-item pair
            normalizer_mat (np.ndarray): Ideal DCG normalization matrix of shape (n_users, k)
                                       for computing normalized scores
            hit (np.ndarray): Binary hit matrix of shape (n_users, k) indicating
                            whether each top-k item is relevant
            ground_truth_rank (np.ndarray): Ranking positions of items in the prediction
                                          of shape (n_users, n_items)
            k (int): Evaluation cutoff - only top-k positions are considered

        Returns:
            float: Mean NDCG@k score across all users
        """
        # Calculate the normalizer first
        normalizer = np.sum(normalizer_mat[:, :k], axis=1)
        # Calculate DCG
        dcg = np.sum(((np.exp2(ratings) - 1) / np.log2(ground_truth_rank + 1)) * hit.astype(float), axis=1)

        avg_dcg_per_user = dcg / normalizer

        return np.mean(avg_dcg_per_user)

    def _NDCG_at_k_array(self, ratings: np.ndarray, normalizer_mat: np.ndarray, hit: np.ndarray, ground_truth_rank: np.ndarray,
                         k: int) -> float:
        """
        Calculate NDCG@k for multiple ranking arrays (e.g., from stochastic reranking).

        This method extends the standard NDCG@k calculation to handle multiple rankings
        per user, typically generated through stochastic reranking procedures. It computes
        the average NDCG@k across all ranking variants for each user.

        Args:
            ratings (np.ndarray): User-item rating matrix of shape (n_users, n_items)
            normalizer_mat (np.ndarray): Ideal DCG normalization matrix
            hit (np.ndarray): Binary hit matrix of shape (n_users, n_rerankings, k)
                            indicating relevance across multiple rankings
            ground_truth_rank (np.ndarray): Ranking positions across multiple rankings
                                          of shape (n_users, n_rerankings, n_items)
            k (int): Evaluation cutoff for top-k positions

        Returns:
            float: Mean NDCG@k score averaged across users and ranking variants
        """
        # Repeat arrays along the new axis to match the dimensionality of ground_truth_rank
        ratings = np.repeat(ratings[:, np.newaxis, :], ground_truth_rank.shape[1], axis=1)
        normalizer_mat = np.repeat(normalizer_mat[:, np.newaxis, :], ground_truth_rank.shape[1], axis=1)

        # Calculate the normalizer first
        normalizer = np.sum(normalizer_mat[:, :, :k], axis=2)  # Sum along the third dimension

        # Calculate DCG
        dcg = np.sum(((np.exp2(ratings) - 1) / np.log2(ground_truth_rank + 1)) * hit.astype(float), axis=2)

        # Get the average DCG per user per rerankings
        avg_dcg_per_user = np.mean(dcg / normalizer, axis=1)

        # Compute the mean across users
        return np.mean(avg_dcg_per_user, axis=0)
    
    @staticmethod
    def fair_calculation(topk: list, sorted_indices: np.ndarray, item_ids: np.ndarray, attribute_array: np.ndarray, predictions: np.ndarray, threshold_k: int, fairness_metric: str = 'exp_avg', fairness_loss_type: str = 'mae'):
        """
        Calculate comprehensive fairness metrics across different top-k cutoffs.

        This method evaluates algorithmic fairness by measuring representation disparities
        between sensitive groups (e.g., demographic groups) in recommendation rankings.
        Multiple fairness metrics are supported, each capturing different aspects of
        fair representation in ranked outputs.

        Args:
            topk (list): List of top-k cutoff values to evaluate (e.g., [5, 10, 20])
            sorted_indices (np.ndarray): Item indices sorted by prediction scores
                                       of shape (n_users, n_items)
            item_ids (np.ndarray): Mapping from indices to actual item IDs
                                 of shape (n_users, n_items)
            attribute_array (np.ndarray): Binary sensitive attribute array where 1 indicates
                                        membership in protected group, shape (n_items,)
            predictions (np.ndarray): Model prediction scores of shape (n_users, n_items)
            threshold_k (int): Threshold cutoff for certain fairness metrics (e.g., sigmoid_thresh)
            fairness_metric (str): Fairness evaluation method:
                                 - 'exp_norm': Exponential normalization of predictions
                                 - 'count': Simple group count differences
                                 - 'sigmoid': Sigmoid-transformed predictions
                                 - 'rank_topk': Average ranking positions
                                 - 'ndcg_diff': NDCG-based group differences
            fairness_loss_type (str): Type of fairness loss calculation:
                                     - 'mae': Mean Absolute Error (default)
                                     - 'mse': Mean Squared Error

        Returns:
            tuple: (fairness_ratio_loss, fairness_loss) where:
                  - fairness_ratio_loss (dict): Ratio-based fairness metrics by top-k
                  - fairness_loss (dict): Absolute difference fairness metrics by top-k
        """
        predictions = np.array(predictions)
        np.set_printoptions(threshold=10000)
        sorted_indices = np.array(sorted_indices)
        fairness_ratio_loss = dict()
        fairness_mae_loss = dict()  # MAE loss
        fairness_mse_loss = dict()  # MSE loss
        variance_group_loss = dict()
        avg_group_ratio = {}  # used for fa*ir
        epsilon = TOLERANCE_EPSILON

        for k in topk:
            num_rows = sorted_indices.shape[0]
            group_a_count = np.array([0] * num_rows)
            group_b_count = np.array([0] * num_rows)
            group_a_norm = np.array([0.0] * num_rows)
            group_b_norm = np.array([0.0] * num_rows)
            selected_attribute_indices = np.take_along_axis(item_ids, sorted_indices[:, :k], axis=1) - 1
            selected_attributes = attribute_array[selected_attribute_indices]

            if fairness_metric == 'exp_norm':
                selected_predictions = np.exp(np.take_along_axis(predictions, sorted_indices[:, :k], axis=1))
                selected_predictions_norm = selected_predictions / selected_predictions.sum(axis=1, keepdims=True)
                group_a_count = selected_attributes.sum(axis=1)
                group_b_count = k - group_a_count
                sum_temp = (selected_attributes * selected_predictions_norm).sum(axis=1)
                group_a_count_temp = np.copy(group_a_count)
                group_b_count_temp = np.copy(group_b_count)
                group_a_count_temp[group_a_count_temp == 0] = 1
                group_b_count_temp[group_b_count_temp == 0] = 1
                group_a_norm_temp = sum_temp / group_a_count_temp
                group_b_norm_temp = (1 - sum_temp) / group_b_count_temp
                group_a_norm = np.where(group_a_count == 0, 0, group_a_norm_temp)
                group_b_norm = np.where(group_b_count == 0, 0, group_b_norm_temp)
                fairness_ratio_loss[k] = np.average(group_a_norm / (group_b_norm + epsilon))
                # MSE loss
                fairness_mse_loss[k] = np.average((group_a_norm - group_b_norm) ** 2)
                # MAE loss
                fairness_mae_loss[k] = np.average(np.abs(group_a_norm - group_b_norm))

                diff = group_a_norm - group_b_norm
                num_top_bottom_elements = max(1, int(0.0002 * num_rows))  # 0.02% of the number of rows

                # Get indices sorted by diff values
                diff_sorted_indices = np.argsort(diff)
        # Choose between MAE and MSE based on fairness_loss_type parameter
        if fairness_loss_type.lower() == 'mse':
            fairness_loss = fairness_mse_loss
        else:  # default to MAE
            fairness_loss = fairness_mae_loss

        logging.info(variance_group_loss)
        return fairness_ratio_loss, fairness_loss

    def evaluate_method(self, predictions: np.ndarray, ratings: np.ndarray, topk: list, metrics: list,
                        item_id: np.ndarray, attribute=None) -> Tuple[
        Dict[str, float], Dict[int, float], Dict[int, float]]:
        """
        Comprehensive evaluation method for recommendation models with fairness analysis.

        This method performs multi-faceted evaluation including:
        - Standard ranking metrics (NDCG, MAP) at various cutoffs
        - Fairness metrics across sensitive groups when attributes are provided
        - Baseline comparisons for color-blind recommendation

        The evaluation considers both recommendation quality and fairness, providing
        insights into potential algorithmic bias in the ranking system.

        Args:
            predictions (np.ndarray): Model prediction scores of shape (n_users, n_candidates)
                                    where first column contains ground-truth item scores
            ratings (np.ndarray): User-item relevance ratings of shape (n_users, n_pos_items)
            topk (list): List of top-k cutoff values for evaluation (e.g., [5, 10, 20])
            metrics (list): List of evaluation metrics to compute (e.g., ['NDCG', 'MAP'])
            item_id (np.ndarray): Item identifier array containing both positive and negative items
                                of shape (n_users, n_candidates)
            attribute (pd.DataFrame, optional): Item attribute information containing sensitive
                                              group memberships. If provided, fairness analysis
                                              is performed.

        Returns:
            tuple: Three-element tuple containing:
                - evaluations (Dict[str, float]): Standard ranking metric results (e.g., 'NDCG@10')
                - fairness_ratio_loss (Dict[int, float]): Fairness ratio metrics by top-k
                - fairness_loss (Dict[int, float]): Fairness absolute difference metrics by top-k
        """
        evaluations = dict()
        rerank_evaluations = dict()

        num_of_users, num_pos_items = ratings.shape
        sorted_ratings = -np.sort(-ratings)  # descending order !!
        discounters = np.tile([np.log2(i + 1) for i in range(1, 1 + num_pos_items)], (num_of_users, 1))
        normalizer_mat = (np.exp2(sorted_ratings) - 1) / discounters

        sorted_indices = (-predictions).argsort(axis=1)  # index of sorted predictions (max->min)
        gt_rank = np.array([np.argwhere(sorted_indices == i)[:, 1] + 1 for i in
                            range(num_pos_items)]).T  # rank of the ground-truth (start from 1)
        if attribute is not None:
            np_attribute = attribute['sensitive'].to_numpy()
            fairness_ratio_loss, fairness_loss = self.fair_calculation(topk, sorted_indices, item_id, np_attribute,
                                                                          predictions, self.threshold_k, self.fairness_metric, self.fairness_loss_type)
            logging.info("Original: ")
            logging.info("fairness_ratio_loss_dic: " + str(fairness_ratio_loss))
            logging.info("fairness_loss_dic: " + str(fairness_loss))

        for k in topk:
            hit = (gt_rank <= k)
            for metric in metrics:
                key = '{}@{}'.format(metric, k)
                if metric == 'NDCG':
                    evaluations[key] = self._NDCG_at_k(ratings, normalizer_mat, hit, gt_rank, k)
                elif metric == 'MAP':
                    evaluations[key] = self._MAP_at_k(hit, gt_rank)
                else:
                    raise ValueError('Undefined evaluation metric: {}.'.format(metric))

        logging.info("Original evaluations: " + str(evaluations))
        return evaluations, fairness_ratio_loss, fairness_loss

    def __init__(self, args):
        """
        Initialize the BaseRunner with configuration parameters.

        Sets up all training, evaluation, and fairness-related parameters based on
        command-line arguments or configuration objects.

        Args:
            args: Configuration object containing all necessary parameters for
                 training, evaluation, and fairness assessment
        """
        self.epoch = args.epoch
        self.check_epoch = args.check_epoch
        self.test_epoch = args.test_epoch
        self.early_stop = args.early_stop
        self.early_stop_thresh = args.early_stop_thresh
        self.learning_rate = args.lr
        self.schedule = eval(args.schedule)
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.l2 = args.l2
        self.optimizer_name = args.optimizer
        self.clip_value = args.clip_value
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.topk = [int(x) for x in args.topk.split(',')]
        self.metrics = [m.strip().upper() for m in args.metric.split(',')]
        self.main_metric = '{}@{}'.format(self.metrics[0], self.topk[0])  # early stop based on main_metric
        self.last_metric = '{}@{}'.format(self.metrics[-1], self.topk[-1])  # save model based on last_metric
        self.prefetch_factor = args.prefetch_factor
        self.time = None  # will store [start_time, last_step_time]
        self.model_modes = args.model_modes.split(',')
        self.fair_result_mode = args.fair_result_mode
        self.fairness_metric = args.fairness_metric
        self.baseline = args.baseline
        self.data_p = args.data_p
        self.threshold_k = args.threshold_k
        self.fairness_loss_type = args.fairness_loss_type

    def _adjust_lr(self, optimizer, epoch):
        """
        Adjust learning rate according to the predefined schedule.

        Implements step-wise learning rate decay based on epoch milestones.
        When an epoch reaches a milestone, the learning rate is reduced by a factor of 4.

        Args:
            optimizer: PyTorch optimizer whose learning rate will be adjusted
            epoch (int): Current training epoch number
        """
        lr = self.learning_rate
        for milestone in self.schedule:
            lr *= 0.25 if epoch >= milestone else 1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def _check_time(self, start=False):
        """
        Track and measure training time intervals.

        Provides timing functionality for monitoring training progress and
        measuring time elapsed between checkpoints.

        Args:
            start (bool): If True, initializes timing. If False, returns elapsed time
                        since last check.

        Returns:
            float: Elapsed time in seconds since last check, or start time if start=True
        """
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def _build_optimizer(self, model):
        """
        Construct and configure the optimization algorithm.

        Creates a PyTorch optimizer instance based on the specified optimizer type,
        learning rate, and weight decay parameters. Supports various optimization
        algorithms including Adam, SGD, Adagrad, and Adadelta.

        Args:
            model: PyTorch model whose parameters will be optimized

        Returns:
            torch.optim.Optimizer: Configured optimizer instance
        """
        logging.info('Optimizer: ' + self.optimizer_name)
        optimizer = eval('torch.optim.{}'.format(self.optimizer_name))(
            model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
        return optimizer

    def _fairness_ndcg_calculator(self, norm_fairness, ndcg):
        """
        Calculate combined fairness-utility metric.

        Computes a harmonic-like combination of fairness and NDCG metrics,
        balancing recommendation quality with fairness considerations.

        Args:
            norm_fairness (float): Normalized fairness score (0 = perfectly fair)
            ndcg (float): Normalized Discounted Cumulative Gain score

        Returns:
            float: Combined fairness-utility score
        """
        return (1 - norm_fairness) * ndcg / (1 - norm_fairness + ndcg)

    def train(self, data_dict: Dict[str, BaseModel.Dataset], attribute: pd.DataFrame) -> NoReturn:
        """
        Execute the complete model training process with fairness monitoring.

        Manages the full training lifecycle including:
        - Model fitting across multiple epochs
        - Performance evaluation on development set
        - Fairness metric computation and tracking
        - Early stopping based on accuracy and fairness criteria
        - Model checkpointing and saving
        - Learning rate scheduling

        The training process balances multiple objectives: recommendation accuracy,
        fairness across sensitive groups, and computational efficiency.

        Args:
            data_dict (Dict[str, BaseModel.Dataset]): Dictionary containing training,
                                                    development, and test datasets
            attribute (pd.DataFrame): Item attribute information for fairness evaluation
                                    containing sensitive group memberships

        Returns:
            NoReturn: This method manages the training process but does not return values
        """
        model = data_dict['train'].model
        main_metric_results, fairness_results = list(), list()
        dev_fairness_results = {key: [] for key in self.topk}
        dev_log_fairness_results = {key: [] for key in self.topk}
        dev_fairness_ndcg_results = {key: [] for key in self.topk}
        dev_log_fairness_ndcg_results = {key: [] for key in self.topk}
        print_k = self.topk[0]
        last_k = self.topk[-1]
        self._check_time(start=True)
        try:
            for epoch in range(self.epoch):
                # Fit
                self._check_time()
                gc.collect()
                torch.cuda.empty_cache()
                loss = self.fit(data_dict['train'], epoch=epoch + 1)
                gc.collect()
                torch.cuda.empty_cache()
                training_time = self._check_time()

                # user_num = len(data_dict['dev'].data['user_id'])
                # Observe selected tensors
                if len(model.check_list) > 0 and self.check_epoch > 0 and epoch % self.check_epoch == 0:
                    utils.check(model.check_list)

                # Record dev results
                dev_accuracy_result, dev_fairness_result, dev_log_fairness_result = self.evaluate(data_dict['dev'],
                                                                                                  self.topk,
                                                                                                  self.metrics,
                                                                                                  attribute)
                main_metric_results.append(dev_accuracy_result[self.last_metric])
                fairness_results.append(dev_log_fairness_result[last_k])
                for key, value in dev_fairness_result.items():
                    dev_fairness_results[key].append(value)
                    dev_fairness_ndcg_results[key].append(
                        self._fairness_ndcg_calculator(value / int(key), main_metric_results[-1]))

                for key, value in dev_log_fairness_result.items():
                    dev_log_fairness_results[key].append(value)
                    if self.fair_result_mode == '1':
                        dev_log_fairness_ndcg_results[key].append(
                            self._fairness_ndcg_calculator(value / int(key), main_metric_results[-1]))
                    else:
                        dev_log_fairness_ndcg_results[key].append(
                            self._fairness_ndcg_calculator(value, main_metric_results[-1]))

                logging_str = 'Epoch {:<5} loss={:<.4f} fairness_ndcg={:<.4f} log_fairness_ndcg={:<.4f} [{:<3.1f} s]    dev=({})'.format(
                    epoch + 1, loss, dev_fairness_ndcg_results[print_k][-1], dev_log_fairness_ndcg_results[print_k][-1],
                    training_time, utils.format_metric(dev_accuracy_result))

                # Test
                if self.test_epoch > 0 and epoch % self.test_epoch == 0:
                    test_result, _, _ = self.evaluate(data_dict['test'], self.topk, self.metrics, attribute)
                    logging_str += ' test=({})'.format(utils.format_metric(test_result))
                testing_time = self._check_time()
                logging_str += ' [{:<.1f} s]'.format(testing_time)

                # Save accuracy model
                if max(main_metric_results) == main_metric_results[-1] or min(fairness_results) == fairness_results[-1] or \
                        (hasattr(model, 'stage') and model.stage == 1):
                    model.save_model(model_tag='max')
                    logging_str += ' *'
                
                if (epoch + 1) % 20 == 0 or epoch == 0:
                    model.save_model(model_tag= 'e' + str(epoch + 1))
                    logging_str += ' e'

                # Save fairness model
                for k in self.topk:
                    break
                    if 'f' in self.model_modes:
                        if min(dev_fairness_results[k]) == dev_fairness_results[k][-1] and self.fair_result_mode == '1':
                            model.save_model(model_tag=k, model_mode='f')
                            logging_str += ' * F T' + str(k)
                        if min(dev_log_fairness_results[k]) == dev_log_fairness_results[k][-1]:
                            model.save_model(model_tag=k, model_mode='flog')
                            logging_str += ' * Flog T' + str(k)
                    if 'fn' in self.model_modes:
                        if max(dev_fairness_ndcg_results[k]) == dev_fairness_ndcg_results[k][-1] \
                                and self.fair_result_mode == '1':
                            model.save_model(model_tag=k, model_mode='fn')
                            logging_str += ' * FN T' + str(k)
                        if max(dev_log_fairness_ndcg_results[k]) == dev_log_fairness_ndcg_results[k][-1]:
                            model.save_model(model_tag=k, model_mode='fnlog')
                            logging_str += ' * FNlog T' + str(k)
                    if 'n' in self.model_modes:
                        if max(main_metric_results) == main_metric_results[-1] or \
                                (hasattr(model, 'stage') and model.stage == 1):
                            model.save_model(model_mode='n')
                            logging_str += ' *'
                logging.info(logging_str)

                if self.early_stop > 0 and self.eval_termination(main_metric_results, fairness_results):
                    logging.info("Early stop at %d based on dev result." % (epoch + 1))
                    break
        except KeyboardInterrupt:
            logging.info("Early stop manually")
            exit_here = input("Exit completely without evaluation? (y/n) (default n):")
            if exit_here.lower().startswith('y'):
                logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
                exit(1)
        model.load_model(model_tag='e'+str(self.epoch))

    def _add_ids(self, batch: dict, out_dict: dict) -> dict:
        """
        Extract and transfer batch metadata to output dictionary.

        Transfers essential batch information including user IDs, item IDs, ratings,
        and fairness-related metadata from input batch to the model output dictionary.
        This ensures that downstream evaluation can properly associate predictions
        with their corresponding users, items, and fairness attributes.

        Args:
            batch (dict): Input batch containing user/item data and metadata
            out_dict (dict): Model output dictionary to be augmented with batch metadata

        Returns:
            dict: Enhanced output dictionary containing both model outputs and batch metadata
        """
        out_dict['user_id'] = batch['user_id'].clone()
        out_dict['item_id'] = batch['item_id'].clone()
        out_dict['rating'] = batch['rating'].clone()
        out_dict['num_pos_items'] = batch['num_pos_items'].clone()
        out_dict['ideal_dcg'] = batch['ideal_dcg'].clone()
        out_dict['a_index'] = batch['a_index'].clone()
        out_dict['b_index'] = batch['b_index'].clone()
        out_dict['mask_ratio_a'] = batch['mask_ratio_a'].clone()
        out_dict['mask_ratio_b'] = batch['mask_ratio_b'].clone()
        out_dict['rho'] = batch['rho'].clone()

        return out_dict

    def fit(self, data: BaseModel.Dataset, epoch=-1) -> float:
        """
        Perform one epoch of model training.

        Executes a complete training epoch including forward pass, loss computation,
        backpropagation, and parameter updates. Supports gradient clipping and
        learning rate adjustment.

        Args:
            data (BaseModel.Dataset): Training dataset
            epoch (int): Current epoch number (-1 for no epoch-specific behavior)

        Returns:
            float: Average training loss for the epoch
        """
        model = data.model
        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        data.actions_before_epoch()  # must sample before multi thread start

        self._adjust_lr(model.optimizer, epoch)

        model.train()
        loss_lst = list()
        if self.prefetch_factor != 0:
            dl = DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                        collate_fn=data.collate_batch, pin_memory=self.pin_memory, prefetch_factor=self.prefetch_factor)
        else:
            dl = DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                        collate_fn=data.collate_batch, pin_memory=self.pin_memory)
        for batch in tqdm(dl, leave=False, desc='Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
            batch = utils.batch_to_gpu(batch, model.device)
            model.optimizer.zero_grad()
            out_dict = model(batch)
            out_dict = self._add_ids(batch, out_dict)
            loss = model.loss(out_dict, epoch)
            loss.backward()
            if self.clip_value > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)
            model.optimizer.step()
            loss_lst.append(loss.detach().cpu().data.numpy())

        return np.mean(loss_lst).item()

    def eval_termination(self, criterion: List[float], fairness_results: List[float] = []) -> bool:
        """
        Determine whether to terminate training based on early stopping criteria.

        Evaluates multiple termination conditions:
        - Sustained non-improvement in primary metric
        - Distance from best performance exceeding threshold
        - Combined accuracy and fairness stagnation

        Args:
            criterion (List[float]): History of primary evaluation metric (e.g., NDCG)
            fairness_results (List[float]): History of fairness metric values

        Returns:
            bool: True if training should be terminated, False otherwise
        """
        if not fairness_results:
            if len(criterion) > self.early_stop_thresh and utils.non_increasing(criterion[-self.early_stop:]):
                return True
            elif len(criterion) - criterion.index(max(criterion)) > self.early_stop_thresh:
                return True
        else:
            if len(criterion) > self.early_stop_thresh and utils.non_increasing(criterion[-self.early_stop:]) and \
                    utils.non_decreasing(fairness_results[-self.early_stop:]):
                return True
            elif len(criterion) - criterion.index(max(criterion)) > self.early_stop_thresh and \
                    len(fairness_results) - fairness_results.index(min(fairness_results)) > self.early_stop_thresh:
                return True
        return False

    def evaluate(self, data: BaseModel.Dataset, topks: list, metrics: list, attribute=None) -> Tuple[
        Dict[str, float], Dict[int, int], Dict[int, float]]:
        """
        Evaluate model performance on a given dataset.

        Performs comprehensive evaluation including ranking metrics and fairness analysis.
        This method coordinates prediction generation and metric computation.

        Args:
            data (BaseModel.Dataset): Dataset to evaluate on
            topks (list): List of top-k cutoffs for evaluation
            metrics (list): List of ranking metrics to compute
            attribute (pd.DataFrame, optional): Attribute data for fairness evaluation

        Returns:
            tuple: (ranking_metrics, fairness_ratios, fairness_losses)
        """
        predictions, ratings = self.predict(data)
        return self.evaluate_method(predictions, ratings, topks, metrics,
                                    np.concatenate([data.data['item_id'], data.data['neg_items']], axis=1), attribute)

    def predict(self, data: BaseModel.Dataset) -> Tuple[ndarray, ndarray]:
        """
        Generate model predictions for a dataset.

        Performs inference on the given dataset, producing prediction scores for
        all user-item pairs. The prediction format places ground-truth items first
        followed by negative samples for each user.

        Example:
            Ground-truth items: [1, 2]
            Negative items: [[3,4], [5,6]]
            Predictions: [[pred_1, pred_3, pred_4], [pred_2, pred_5, pred_6]]

        Args:
            data (BaseModel.Dataset): Dataset to generate predictions for

        Returns:
            tuple: (predictions, ratings) where:
                - predictions (ndarray): Shape (n_users, n_items) prediction scores
                - ratings (ndarray): Shape (n_users, n_pos_items) ground truth ratings
        """
        data.model.eval()
        predictions = list()
        ratings = list()
        dl = DataLoader(data, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers,
                        collate_fn=data.collate_batch, pin_memory=self.pin_memory)
        for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
            prediction = data.model(utils.batch_to_gpu(batch, data.model.device))['prediction']
            predictions.extend(prediction.cpu().data.numpy())
            ratings.extend(batch['rating'].cpu().data.numpy())
        return np.array(predictions), np.array(ratings)  # [# of users, # of items], [# of users, # of pos items]

    def print_res(self, data: BaseModel.Dataset, attribute=None) -> str:
        """
        Generate formatted result string for model evaluation.

        Creates a human-readable summary of model performance metrics
        for reporting and logging purposes.

        Args:
            data (BaseModel.Dataset): Dataset to evaluate on
            attribute (pd.DataFrame, optional): Attribute data for fairness metrics

        Returns:
            str: Formatted string containing evaluation results
        """
        accuracy_dict, _, _ = self.evaluate(data, self.topk, self.metrics, attribute)
        res_str = '(' + utils.format_metric(accuracy_dict) + ')'
        return res_str
