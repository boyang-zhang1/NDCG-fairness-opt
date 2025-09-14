# -*- coding: UTF-8 -*-

# Standard library imports
import gc
import json
import logging
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import tempfile
from multiprocessing import Pool
from time import time
from typing import Dict, List, NoReturn, Tuple

# Third-party imports
import cvxpy as cp
import fairsearchcore as fsc
from fairsearchcore.models import FairScoreDoc
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
NUM_RERANKINGS = 100
TOLERANCE_EPSILON = 1e-5
CONVX_TOLERANCE = 1e-15
SOLVER_ABS_TOL = 1e-10
SOLVER_REL_TOL = 1e-10

def sample_ranking(probability_matrix):
    """
    Generate a ranking by sampling from probability matrix.

    Args:
        probability_matrix (np.ndarray): Probability matrix for ranking

    Returns:
        list: Sampled ranking indices
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
    @staticmethod
    def parse_runner_args(parser):
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
                            help='print baseline test result, avaliable: 0: color blind, 1: Demographic Parity Constraints, 2: FA*IR, 3: Feldman et al.')
        parser.add_argument('--data_p', type=float, default=0.28,
                            help='the dataset sensitive group proportion')
        parser.add_argument('--threshold_k', type=int, default=5,
                            help='The number of threshold_k items during fair evaluation.')
        return parser

    def _MAP_at_k(self, hit: np.ndarray, ground_truth_rank: np.ndarray) -> float:
        """
        Calculate Mean Average Precision at k.

        Args:
            hit (np.ndarray): Binary hit matrix
            ground_truth_rank (np.ndarray): Ground truth ranking positions

        Returns:
            float: MAP@k score
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
        Calculate Normalized Discounted Cumulative Gain at k.

        Args:
            ratings (np.ndarray): Rating matrix
            normalizer_mat (np.ndarray): Normalization matrix
            hit (np.ndarray): Binary hit matrix
            ground_truth_rank (np.ndarray): Ground truth ranking positions
            k (int): Top-k value

        Returns:
            float: NDCG@k score
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
        Calculate NDCG at k for array of rankings.

        Args:
            ratings (np.ndarray): Rating matrix
            normalizer_mat (np.ndarray): Normalization matrix
            hit (np.ndarray): Binary hit matrix
            ground_truth_rank (np.ndarray): Ground truth ranking positions
            k (int): Top-k value

        Returns:
            float: NDCG@k score
        """
        # Repeat arrays along the new axis to match the dimensionality of ground_truth_rank
        ratings = np.repeat(ratings[:, np.newaxis, :], ground_truth_rank.shape[1], axis=1)
        normalizer_mat = np.repeat(normalizer_mat[:, np.newaxis, :], ground_truth_rank.shape[1], axis=1)

        # Calculate the normalizer first
        normalizer = np.sum(normalizer_mat[:, :, :k], axis=2)  # Sum along the third dimension

        # Calculate DCG
        dcg = np.sum(((np.exp2(ratings) - 1) / np.log2(ground_truth_rank + 1)) * hit.astype(float), axis=2)

        # Get the average DCG per user per NUM_RERANKINGS rerankings
        avg_dcg_per_user = np.mean(dcg / normalizer, axis=1)

        # Compute the mean across users
        return np.mean(avg_dcg_per_user, axis=0)
    
    @staticmethod
    def fair_calculation(topk: list, sorted_indices: np.ndarray, item_ids: np.ndarray, attribute_array: np.ndarray, predictions: np.ndarray, threshold_k: int, fairness_metric: str = 'exp_avg'):
        """
        Calculate fairness metrics for different top-k values.

        Args:
            topk (list): List of top-k values to evaluate
            sorted_indices (np.ndarray): Sorted item indices by predictions
            item_ids (np.ndarray): Item ID array
            attribute_array (np.ndarray): Binary attribute array (sensitive groups)
            predictions (np.ndarray): Model predictions
            threshold_k (int): Threshold k for certain fairness metrics
            fairness_metric (str): Type of fairness metric to calculate

        Returns:
            tuple: (fairness_ratio_loss, fairness_loss) dictionaries
        """
        predictions = np.array(predictions)
        np.set_printoptions(threshold=10000)
        sorted_indices = np.array(sorted_indices)
        fairness_ratio_loss = dict()
        fairness_loss = dict()  # MAE loss
        fairness_loss = dict()  # MSE loss
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

                diff = group_a_norm - group_b_norm
                num_top_bottom_elements = max(1, int(0.0002 * num_rows))  # 0.02% of the number of rows

                # Get indices sorted by diff values
                diff_sorted_indices = np.argsort(diff)
                
                # Calculate cross entropy loss for analysis (k == 305)
                if k == 305:
                    cross_entropy_loss = -np.sum(np.log(selected_predictions_norm + epsilon), axis=1)
                    avg_cross_entropy = np.mean(cross_entropy_loss)

            elif fairness_metric == 'exp_norm_topk':
                exp_predictions = np.exp(predictions)
                # Normalize these exponentiated values
                exp_predictions_norm = exp_predictions / exp_predictions.sum(axis=1, keepdims=True)
                # Assuming the ranked index remains the same, we sort and then take the top-k
                selected_predictions_norm = np.take_along_axis(exp_predictions_norm, sorted_indices[:, :k], axis=1)
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
                fairness_loss[k] = np.average(np.abs(group_a_norm - group_b_norm))                

            elif fairness_metric == 'rank_topk':
                # Generate a 1D rank list for all users
                rank_list = np.arange(1, k + 1)
                rank_fallback = k + 1 

                mask_group_a = selected_attributes == 1
                mask_group_b = ~mask_group_a  # Assuming binary attributes, so directly using negation

                sum_group_a = np.dot(mask_group_a, rank_list)
                sum_group_b = np.dot(mask_group_b, rank_list)
                count_group_a = np.sum(mask_group_a, axis=-1)
                count_group_b = np.sum(mask_group_b, axis=-1)

                avg_group_a = np.zeros(selected_attributes.shape[0])
                avg_group_b = np.zeros(selected_attributes.shape[0])

                # Only compute average where count is non-zero to avoid division by zero
                non_zero_a = count_group_a != 0
                non_zero_b = count_group_b != 0

                avg_group_a[non_zero_a] = sum_group_a[non_zero_a] / count_group_a[non_zero_a]
                avg_group_b[non_zero_b] = sum_group_b[non_zero_b] / count_group_b[non_zero_b]

                # Apply fallback value where count is zero
                avg_group_a[~non_zero_a] = rank_fallback
                avg_group_b[~non_zero_b] = rank_fallback
                fairness_ratio_loss[k] = np.average(avg_group_a / (avg_group_b + epsilon))
                fairness_loss[k] = np.average(np.abs(avg_group_a - avg_group_b))
            
            elif fairness_metric == 'log_rank_topk':
                # Generate a 1D rank list for all users
                rank_list = 1 / np.log2(1 + np.arange(1, k + 1))
                rank_fallback = 1 / np.log2(1 + k + 1)

                mask_group_a = selected_attributes == 1
                mask_group_b = ~mask_group_a  # Assuming binary attributes, so directly using negation

                sum_group_a = np.dot(mask_group_a, rank_list)
                sum_group_b = np.dot(mask_group_b, rank_list)
                count_group_a = np.sum(mask_group_a, axis=-1)
                count_group_b = np.sum(mask_group_b, axis=-1)

                avg_group_a = np.zeros(selected_attributes.shape[0])
                avg_group_b = np.zeros(selected_attributes.shape[0])

                # Only compute average where count is non-zero to avoid division by zero
                non_zero_a = count_group_a != 0
                non_zero_b = count_group_b != 0

                avg_group_a[non_zero_a] = sum_group_a[non_zero_a] / count_group_a[non_zero_a]
                avg_group_b[non_zero_b] = sum_group_b[non_zero_b] / count_group_b[non_zero_b]

                # Apply fallback value where count is zero
                avg_group_a[~non_zero_a] = rank_fallback
                avg_group_b[~non_zero_b] = rank_fallback
                fairness_ratio_loss[k] = np.average(avg_group_a / (avg_group_b + epsilon))
                fairness_loss[k] = np.average(np.abs(avg_group_a - avg_group_b))

            elif fairness_metric == 'count':
                group_a_count = selected_attributes.sum(axis=1)
                group_b_count = k - group_a_count
                fairness_ratio_loss[k] = np.average(group_a_count / (group_b_count + epsilon))
                fairness_loss[k] = np.average(np.abs(group_a_count - group_b_count) / k)
            
            elif fairness_metric == 'sigmoid':
                # Apply sigmoid function to predictions
                selected_predictions = 1 / (1 + np.exp(-np.take_along_axis(predictions, sorted_indices[:, :k], axis=1)))
                
                # Calculate the sums for groups A and B
                group_a_count = selected_attributes.sum(axis=1)
                group_b_count = k - group_a_count
                sum_group_a = (selected_attributes * selected_predictions).sum(axis=1)
                sum_group_b = ((1 - selected_attributes) * selected_predictions).sum(axis=1)

                # Calculate normalized values for groups A and B
                group_a_count_temp = np.copy(group_a_count)
                group_b_count_temp = np.copy(group_b_count)
                group_a_count_temp[group_a_count_temp == 0] = 1
                group_b_count_temp[group_b_count_temp == 0] = 1
                group_a_norm = np.where(group_a_count == 0, 0, sum_group_a / group_a_count_temp)
                group_b_norm = np.where(group_b_count == 0, 0, sum_group_b / group_b_count_temp)

                # Calculate average differences
                fairness_ratio_loss[k] = np.average(group_a_norm / (group_b_norm + epsilon))
                fairness_loss[k] = np.average(np.abs(group_a_norm - group_b_norm))
            
            elif fairness_metric == 'sigmoid_thresh':
                # Adjust predictions by subtracting the k-th score as a threshold
                threshold = np.take_along_axis(predictions, sorted_indices[:, threshold_k-1:threshold_k], axis=1)
                adjusted_predictions = np.take_along_axis(predictions, sorted_indices[:, :k], axis=1) - threshold

                # Apply sigmoid function to the adjusted predictions
                selected_predictions = 1 / (1 + np.exp(-adjusted_predictions))

                # Calculate the sums for groups A and B
                group_a_count = selected_attributes.sum(axis=1)
                group_b_count = k - group_a_count
                sum_group_a = (selected_attributes * selected_predictions).sum(axis=1)
                sum_group_b = ((1 - selected_attributes) * selected_predictions).sum(axis=1)

                # Calculate normalized values for groups A and B
                group_a_count_temp = np.copy(group_a_count)
                group_b_count_temp = np.copy(group_b_count)
                group_a_count_temp[group_a_count_temp == 0] = 1
                group_b_count_temp[group_b_count_temp == 0] = 1
                group_a_norm = np.where(group_a_count == 0, 0, sum_group_a / group_a_count_temp)
                group_b_norm = np.where(group_b_count == 0, 0, sum_group_b / group_b_count_temp)

                # Calculate average differences
                fairness_ratio_loss[k] = np.average(group_a_norm / (group_b_norm + epsilon))
                fairness_loss[k] = np.average(np.abs(group_a_norm - group_b_norm))

            elif fairness_metric == 'ndcg_diff':
                # Extract sorted predictions and attributes for the top k items
                sorted_predictions = np.take_along_axis(predictions, sorted_indices[:, :k], axis=1)
                selected_attributes = np.take_along_axis(np_attribute[item_id - 1], sorted_indices[:, :k], axis=1)

                # Calculate DCG for current sorting
                log_positions = np.log2(np.arange(2, k + 2))
                gt_a = selected_attributes.astype(float)  # Ground truth where G1 is on top
                dcg_a = np.sum((np.power(2, gt_a) - 1) / log_positions, axis=1)
                gt_b = (~selected_attributes).astype(float)  # Ground truth where G2 is on top
                dcg_b = np.sum((np.power(2, gt_b) - 1) / log_positions, axis=1)

                # Calculate IDCG for group A
                ideal_sorted_a = np.sort(selected_attributes, axis=1)[:, ::-1]  # Sort group A items to the top
                idcg_a = np.sum((np.power(2, ideal_sorted_a) - 1) / log_positions, axis=1)

                # Calculate IDCG for group B
                ideal_sorted_b = np.sort(~selected_attributes, axis=1)[:, ::-1]  # Sort group B items to the top
                idcg_b = np.sum((np.power(2, ideal_sorted_b) - 1) / log_positions, axis=1)

                # Avoid division by zero
                idcg_a[idcg_a == 0] = 1
                idcg_b[idcg_b == 0] = 1

                # Calculate NDCG for groups A and B
                ndcg_a_top = dcg_a / idcg_a
                ndcg_b_top = dcg_b / idcg_b

                fairness_ratio_loss[k] = np.average(ndcg_a_top / (ndcg_b_top + epsilon))
                # MAE loss
                # fairness_loss[k] = np.average(np.abs(ndcg_a_top - ndcg_b_top))
                # MSE loss
                fairness_mse_loss[k] = np.average((ndcg_a_top - ndcg_b_top) ** 2)

        logging.info(variance_group_loss)
        return fairness_ratio_loss, fairness_loss

            

    @staticmethod
    def parallel_solve1(args):
        """
        Parallel optimization solver for demographic parity constraints.

        Args:
            args: Tuple containing user_idx, user_sorted_indices, predictions, item_id,
                  np_attribute, sorted_indices, gt_rank

        Returns:
            tuple: (reranked_gt_rank, reranked_sorted_indices, avg_move, no_solution, equal_array)
        """
        user_idx, user_sorted_indices, predictions, item_id, np_attribute, sorted_indices, gt_rank = args
        no_solution = 0
        equal_array = 0
        user_scores = predictions[user_idx, user_sorted_indices]
        user_sensitive = np_attribute[np.take(item_id, user_sorted_indices) - 1]

        # Calculate v for the current user
        user_v = np.array([1.0 / (np.log(2 + i)) for i, _ in enumerate(user_scores)])

        # Create and solve the optimization problem using the demographic parity constraint
        P = cp.Variable((len(user_scores), len(user_scores)))
        objective = cp.Maximize(cp.matmul(cp.matmul(user_scores, P), user_v))

        constraints = [
            cp.matmul(np.ones((1, len(user_scores))), P) == np.ones((1, len(user_scores))),
            cp.matmul(P, np.ones((len(user_scores),))) == np.ones((len(user_scores),)),
            0 <= P, P <= 1
        ]

        group1_indices = np.where(user_sensitive)[0]
        group2_indices = np.where(user_sensitive == False)[0]

        group1_weights = np.array([1 / user_scores[group1_indices].sum() if i in group1_indices else 0 for i in
                                   range(len(user_scores))])
        group2_weights = np.array([-1 / user_scores[group2_indices].sum() if i in group2_indices else 0 for i in
                                   range(len(user_scores))])

        demographic_parity_weights = group1_weights + group2_weights
        tolerance = CONVX_TOLERANCE
        constraint_demographic_parity1 = cp.matmul(cp.matmul(demographic_parity_weights, P), user_v) <= tolerance
        constraint_demographic_parity2 = cp.matmul(cp.matmul(demographic_parity_weights, P), user_v) >= -tolerance
        constraints.append(constraint_demographic_parity1)
        constraints.append(constraint_demographic_parity2)

        prob = cp.Problem(objective, constraints)
        solver_options = {
            'abstol': SOLVER_ABS_TOL,  # Absolute tolerance
            'reltol': SOLVER_REL_TOL,  # Relative tolerance
        }
        reranked_gt_rank = np.zeros((NUM_RERANKINGS, len(gt_rank)), dtype=int)
        reranked_sorted_indices = np.zeros((NUM_RERANKINGS, len(sorted_indices)), dtype=int)
        try:
            result = prob.solve(verbose=False, solver=cp.ECOS, **solver_options)
            if P and P.value is not None:
                P_value = np.clip(P.value, 0, 1)

                # Rerank the user's items
                new_ranking = np.zeros((NUM_RERANKINGS, len(sorted_indices)), dtype=int)

                for i in range(NUM_RERANKINGS):
                    new_ranking[i] = sample_ranking(P_value)

                    # Rerank the user's items
                for i in range(NUM_RERANKINGS):
                    reranked_sorted_indices[i] = np.array([np.where(new_ranking[i] == x)[0][0] for x in sorted_indices])
                    reranked_gt_rank[i] = np.array([np.where(new_ranking[i] == x - 1)[0][0] + 1 for x in gt_rank])
            else:
                no_solution = 1
                for i in range(NUM_RERANKINGS):
                    reranked_gt_rank[i] = gt_rank
                    reranked_sorted_indices[i] = sorted_indices
        except Exception as error:
            logging.info("reranked_gt_rank error: " + str(error))
            for i in range(NUM_RERANKINGS):
                reranked_gt_rank[i] = gt_rank
                reranked_sorted_indices[i] = sorted_indices
        avg_move = np.average(np.average(np.abs(reranked_gt_rank - gt_rank)))
        if np.array_equal(gt_rank, reranked_gt_rank[0]):
            equal_array = 1
        return reranked_gt_rank, reranked_sorted_indices, avg_move, no_solution, equal_array

    @staticmethod
    def parallel_solve2(args):
        """
        Parallel solver for FA*IR fairness algorithm.

        Args:
            args: Tuple containing k_value, p, alpha, unfair_rankings, gt_rank, sorted_indices,
                  topk, item_id, np_attribute, num_pos_items, metrics, ratings, normalizer_mat,
                  fairness_metric, predictions

        Returns:
            tuple: (tag, fairness_ratio_loss, fairness_loss, NDCG_results)
        """
        # create the Fair object for each combination of p and alpha
        k_value, p, alpha, unfair_rankings, gt_rank, sorted_indices, topk, item_id, np_attribute, num_pos_items, metrics, ratings, normalizer_mat, fairness_metric, predictions = args
        fair = fsc.Fair(k_value, p, alpha)
        # print(k_value, p, alpha)
        reranks = [fair.re_rank(sorted(unfair_ranking, key=lambda x: x.score, reverse=True)) for unfair_ranking in unfair_rankings]

        reranked_idx = np.array([[item.id for item in sublist] for sublist in reranks])
        repredictions = [[0 for _ in range(len(predictions[0]))] for _ in range(len(predictions))]
        for i in range(len(predictions)):
            for j in range(len(predictions[i])):
                repredictions[i][reranked_idx[i][j]] = predictions[i][sorted_indices[i][j]]

        fairness_ratio_loss, fairness_loss = BaseRunner.fair_calculation(topk, reranked_idx, item_id, np_attribute, repredictions, num_pos_items, fairness_metric)
        reranked_gt_rank = np.array([[np.where(j == i)[0][0] + 1 for i in range(num_pos_items)] for j in reranked_idx])
        NDCG_results = {}
        for k in topk:
            hit = (reranked_gt_rank <= k)
            for metric in metrics:
                key = '{}@{}'.format(metric, k)
                if metric == 'NDCG':
                    NDCG_results[key] = BaseRunner._NDCG_at_k(ratings, normalizer_mat, hit, reranked_gt_rank, k)
        tag = [k_value, p, alpha]
        logging.info(json.dumps({'tag': tag, 'fairness_ratio_loss': fairness_ratio_loss, 'fairness_loss': fairness_loss, 'NDCG_results': NDCG_results}))
        return tag, fairness_ratio_loss, fairness_loss, NDCG_results

    @staticmethod
    def parallel_solve3(args):
        """
        Parallel solver for BlackBoxAuditing repair algorithm.

        Args:
            args: Tuple containing user_sorted_indices, predictions, item_id, np_attribute, ratio

        Returns:
            np.ndarray: Reranked indices
        """
        user_sorted_indices, predictions, item_id, np_attribute, ratio = args
        tempdir = tempfile.mkdtemp()
        input_filename = os.path.join(tempdir, "test.csv")
        output_filename = os.path.join(tempdir, "repaired.csv")

        user_scores = predictions[user_sorted_indices]
        user_sensitive = np_attribute[np.take(item_id, user_sorted_indices) - 1]
        # Sort and return the topN items
        df = pd.DataFrame({'score': user_scores, 'sensitive': user_sensitive})
        df.to_csv(input_filename, index=False)
        subprocess.run(["BlackBoxAuditing-repair", input_filename, output_filename, str(ratio), "True", "-p", "sensitive"],
                       check=True)
        repaired_df = pd.read_csv(output_filename)
        ranklist = repaired_df.sort_values(by='score', ascending=False).index.tolist()

        reranked_idx = np.take(user_sorted_indices, ranklist)

        shutil.rmtree(tempdir)
        return reranked_idx


    def evaluate_method(self, predictions: np.ndarray, ratings: np.ndarray, topk: list, metrics: list,
                        item_id: np.ndarray, attribute=None) -> Tuple[
        Dict[str, float], Dict[int, float], Dict[int, float]]:
        """
        :param predictions: (-1, n_candidates) shape, the first column is the score for ground-truth item
        :param ratings: (# of users, # of pos items)
        :param topk: top-K value list
        :param metrics: metric string list
        :param item_id: attribute list (pos + neg)
        :param attribute: attribute info df
        :return: a result dict, the keys are metric@topk
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
                                                                          predictions, self.threshold_k, self.fairness_metric)
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

        if self.baseline == 1:
            args = [(user_idx, user_sorted_indices, predictions, item_id[user_idx], np_attribute, sorted_indices[user_idx],
                     gt_rank[user_idx]) for user_idx, user_sorted_indices in enumerate(sorted_indices)]

            with Pool() as p:
                results = p.map(BaseRunner.parallel_solve1, args)
            reranked_gt_rank, reranked_sorted_indices, avg_move, no_solution, equal_array = zip(*results)
            reranked_gt_rank = np.array(reranked_gt_rank)
            reranked_sorted_indices = np.array(reranked_sorted_indices)
            avg_move = np.array(avg_move)
            no_solution = np.array(no_solution)
            equal_array = np.array(equal_array)
            fairness_ratio_loss_arr = np.zeros((NUM_RERANKINGS, len(topk)))
            fairness_loss_arr = np.zeros((NUM_RERANKINGS, len(topk)))
            for i in range(reranked_sorted_indices.shape[1]):
                each_reranked_sorted_indices = reranked_sorted_indices[:, i, :]
                repredictions = [[0 for _ in range(len(predictions[0]))] for _ in range(len(predictions))]
                for ii in range(len(predictions)):
                    for j in range(len(predictions[ii])):
                        repredictions[ii][each_reranked_sorted_indices[ii][j]] = predictions[ii][sorted_indices[ii][j]]
                fairness_ratio_loss, fairness_loss = self.fair_calculation(topk, each_reranked_sorted_indices,
                                                                        item_id, np_attribute, repredictions,
                                                                        self.threshold_k, self.fairness_metric)
                fairness_ratio_loss_arr[i] = [fairness_ratio_loss[key] for key in topk]
                fairness_loss_arr[i] = [fairness_loss[key] for key in topk]
            fairness_ratio_loss_arr = np.mean(fairness_ratio_loss_arr, axis=0)
            fairness_ratio_loss_dic = {key: value for key, value in zip(topk, fairness_ratio_loss_arr)}
            fairness_loss_arr = np.mean(fairness_loss_arr, axis=0)
            fairness_loss_dic = {key: value for key, value in zip(topk, fairness_loss_arr)}
            logging.info("Baseline1: ")
            logging.info("fairness_ratio_loss_dic: " + str(fairness_ratio_loss_dic))
            logging.info("fairness_loss_dic: " + str(fairness_loss_dic))
            logging.info("avg_move: " + str(np.average(avg_move)))
            logging.info("equal_array: " + str(np.average(equal_array)))
            logging.info("no_solution: " + str(np.average(no_solution)))

            for k in topk:
                hit = (reranked_gt_rank <= k)
                for metric in metrics:
                    key = '{}@{}'.format(metric, k)
                    if metric == 'NDCG':
                        rerank_evaluations[key] = self._NDCG_at_k_array(ratings, normalizer_mat, hit,
                                                                        reranked_gt_rank, k)
                    # elif metric == 'MAP':
                    #     rerank_evaluations[key] = self._MAP_at_k(hit, reranked_gt_rank[:20])
                    else:
                        raise ValueError('Undefined evaluation metric: {}.'.format(metric))
            logging.info("baseline1 evaluations: " + str(rerank_evaluations))

        if self.baseline == 2:
            k_value = len(sorted_indices[0])  # number of topK elements returned (value should be between 10 and 400)
            p_min = max(self.data_p - 0.1, 0)
            p_max = min(self.data_p + 0.1, 1)
            p_values = np.arange(p_min, p_max + 0.001, 0.1)  # create array of p values from 0.02 to 0.98 with step size 0.01
            alpha_values = np.arange(0.01, 0.15,
                                     0.065)  # create array of alpha values from 0.01 to 0.15 with step size 0.01
            unfair_rankings = []
            for i in range(len(predictions)):
                unfair_ranking = []
                u_attribute = np_attribute[item_id[i] - 1]
                for j in range(len(predictions[0])):
                    unfair_ranking.append(FairScoreDoc(j, predictions[i][j], u_attribute[j]))
                unfair_rankings.append(unfair_ranking)

            args = [(k_value, p, alpha, unfair_rankings, gt_rank, sorted_indices, topk, item_id, np_attribute, num_pos_items, metrics, ratings,
                     normalizer_mat, self.fairness_metric, predictions) for p in p_values for alpha in alpha_values]

            with Pool() as p:
                results = p.map(BaseRunner.parallel_solve2, args)
            # args = (k_value, 0.2, 0.1, unfair_rankings, gt_rank, sorted_indices, topk, item_id, np_attribute, num_pos_items, metrics, ratings, normalizer_mat)
            # results = BaseRunner.parallel_solve2(args)
            tag, fairness_ratio_loss, fairness_loss, NDCG_results = zip(*results)
            tag = np.array(tag)
            fairness_ratio_loss = np.array(fairness_ratio_loss)
            fairness_loss = np.array(fairness_loss)
            # NDCG_results = np.array(NDCG_results)
            # for i in range(len(tag)):
            #     print(tag[i], fairness_ratio_loss[i], fairness_loss[i], NDCG_results[i])

        if self.baseline == 3:
            ratios = np.arange(0.1, 1, 0.4)
            for ratio in ratios:
                args = [(sorted_indices[i], predictions[i], item_id[i], np_attribute, ratio) for i in range(len(predictions))]
                # args = [(sorted_indices[i], predictions[i], item_id[i], np_attribute, ratio) for i in range(10)]
                with Pool() as p:
                    results = p.map(BaseRunner.parallel_solve3, args)
                reranked_idx = np.array(results)
                repredictions = [[0 for _ in range(len(predictions[0]))] for _ in range(len(predictions))]
                for i in range(len(predictions)):
                    for j in range(len(predictions[i])):
                        repredictions[i][reranked_idx[i][j]] = predictions[i][sorted_indices[i][j]]
                fairness_ratio_loss, fairness_loss = BaseRunner.fair_calculation(topk, reranked_idx, item_id, np_attribute,
                                                                              repredictions, self.threshold_k, self.fairness_metric)
                NDCG_results = {}
                reranked_gt_rank = np.zeros((len(gt_rank), len(gt_rank[0])), dtype=int)
                for i in range(len(gt_rank)):
                    reranked_gt_rank[i] = np.array([np.where(reranked_idx[i] == x)[0][0] + 1 for x in np.take(sorted_indices[i], gt_rank[i] - 1)])
                for k in topk:
                    hit = (reranked_gt_rank <= k)
                    for metric in metrics:
                        key = '{}@{}'.format(metric, k)
                        if metric == 'NDCG':
                            NDCG_results[key] = BaseRunner._NDCG_at_k(ratings, normalizer_mat, hit, reranked_gt_rank, k)
                logging.info(json.dumps({'ratio': ratio, 'fairness_ratio_loss': fairness_ratio_loss, 'fairness_loss': fairness_loss,
                                         'NDCG_results': NDCG_results}))
        return evaluations, fairness_ratio_loss, fairness_loss

    def __init__(self, args):
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

    def _adjust_lr(self, optimizer, epoch):
        lr = self.learning_rate
        for milestone in self.schedule:
            lr *= 0.25 if epoch >= milestone else 1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def _build_optimizer(self, model):
        logging.info('Optimizer: ' + self.optimizer_name)
        optimizer = eval('torch.optim.{}'.format(self.optimizer_name))(
            model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
        return optimizer

    def _fairness_ndcg_calculator(self, norm_fairness, ndcg):
        return (1 - norm_fairness) * ndcg / (1 - norm_fairness + ndcg)

    def train(self, data_dict: Dict[str, BaseModel.Dataset], attribute: pd.DataFrame) -> NoReturn:
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
            extract user ids and item ids from a batch
            and add them into out_dict
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
        Evaluate the results for an input dataset.
        :return: result dict (key: metric@k)
        """
        predictions, ratings = self.predict(data)
        return self.evaluate_method(predictions, ratings, topks, metrics,
                                    np.concatenate([data.data['item_id'], data.data['neg_items']], axis=1), attribute)

    def predict(self, data: BaseModel.Dataset) -> Tuple[ndarray, ndarray]:
        """
        The returned prediction is a 2D-array, each row corresponds to all the candidates,
        and the ground-truth item poses the first.
        Example: ground-truth items: [1, 2], 2 negative items for each instance: [[3,4], [5,6]]
                 predictions like: [[1,3,4], [2,5,6]]
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
        Construct the final result string before/after training
        :return: test result string
        """
        accuracy_dict, _, _ = self.evaluate(data, self.topk, self.metrics, attribute)
        res_str = '(' + utils.format_metric(accuracy_dict) + ')'
        return res_str
