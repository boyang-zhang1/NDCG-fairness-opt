# -*- coding: UTF-8 -*-

import time

import pandas as pd
import torch
import logging
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as BaseDataset
from torch.nn.utils.rnn import pad_sequence
from typing import NoReturn, List

from utils import utils
from helpers.BaseReader import BaseReader
from . import losses, ndcg_loss

# Global variable to store item attributes for fairness calculations
ItemAttributes = {}

def calculate_ideal_dcg(ratings, top_k=-1):
    """
    Calculate the Ideal Discounted Cumulative Gain (IDCG) for a list of ratings.

    Args:
        ratings (list): List of relevance ratings
        top_k (int): Number of top items to consider. -1 means all items

    Returns:
        float: The ideal DCG value
    """
    ratings_sorted = -np.sort(-np.array(ratings))  # Sort in descending order
    position_weights = np.log2(1.0 + np.arange(1, len(ratings) + 1))
    dcg_size = top_k if top_k != -1 else len(ratings)
    ideal_dcg = np.sum(((2 ** ratings_sorted - 1) / position_weights)[:dcg_size])
    return ideal_dcg



class BaseModel(nn.Module):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = []

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--model_path', type=str, default='',
                            help='Model save path.')
        parser.add_argument('--buffer', type=int, default=1,
                            help='Whether to buffer feed dicts for dev/test')
        parser.add_argument('--reorg_train_data', type=int, default=0,
                            help='Whether to reorganize the training data')
        return parser

    @staticmethod
    def init_weights(m):
        if 'Linear' in str(type(m)):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif 'Embedding' in str(type(m)):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def __init__(self, args, corpus: BaseReader):
        super(BaseModel, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_path = args.model_path
        self.buffer = args.buffer
        self.reorg_train_data = args.reorg_train_data
        self.optimizer = None
        self.check_list = list()  # List of tensors to observe during training

        self._define_params()
        self.total_parameters = self.count_variables()
        logging.info(f'Model parameters: {self.total_parameters}')

    """
    Key Methods
    """
    def _define_params(self) -> NoReturn:
        """
        Define model parameters. To be implemented by subclasses.
        """
        pass

    def forward(self, feed_dict: dict) -> dict:
        """
        Forward pass of the model.

        Args:
            feed_dict (dict): Batch data prepared in Dataset

        Returns:
            dict: Output dictionary including prediction with shape [batch_size, n_candidates]
        """
        pass

    def loss(self, out_dict: dict, attributes) -> torch.Tensor:
        """
        Calculate loss for the model.

        Args:
            out_dict (dict): Model output dictionary
            attributes: Item attributes for loss calculation

        Returns:
            torch.Tensor: Computed loss value
        """
        pass

    """
    Auxiliary Methods
    """
    def customize_parameters(self) -> list:
        """
        Customize optimizer settings for different parameter groups.
        Separates weight and bias parameters for different weight decay settings.

        Returns:
            list: List of parameter groups with custom settings
        """
        weight_params, bias_params = [], []
        for name, param in filter(lambda x: x[1].requires_grad, self.named_parameters()):
            if 'bias' in name:
                bias_params.append(param)
            else:
                weight_params.append(param)

        parameter_groups = [
            {'params': weight_params},
            {'params': bias_params, 'weight_decay': 0}
        ]
        return parameter_groups

    def _model_path(self, model_path, model_tag) -> str:
        if model_path is None:
            if model_tag is None:
                model_path = self.model_path
            else:
                separator = '.'
                path_split = self.model_path.split(separator)
                model_path = separator.join(path_split[:-1]) + '_tag_' + str(model_tag) + separator + path_split[-1]
        return model_path

    def save_model(self, model_path=None, model_tag=None) -> NoReturn:
        model_path = self._model_path(model_path, model_tag)
        utils.check_dir(model_path)
        torch.save(self.state_dict(), model_path)

    def load_model(self, model_path=None, model_tag=None) -> NoReturn:
        model_path = self._model_path(model_path, model_tag)
        state_dict = torch.load(model_path, map_location=self.device)
        
        # Handle size mismatch for item embeddings during model loading
        if 'mlp_i_embeddings.weight' in state_dict:
            pretrained_embeddings = state_dict['mlp_i_embeddings.weight']
            current_embeddings = self.mlp_i_embeddings.weight.data

            if pretrained_embeddings.size(0) != current_embeddings.size(0):
                logging.warning(
                    f"Size mismatch for mlp_i_embeddings.weight: "
                    f"pretrained size {pretrained_embeddings.size()}, "
                    f"current size {current_embeddings.size()}. "
                    "Initializing additional embeddings randomly."
                )

                # Initialize additional embeddings with normal distribution
                num_additional = current_embeddings.size(0) - pretrained_embeddings.size(0)
                embedding_dim = pretrained_embeddings.size(1)
                additional_embeddings = nn.init.normal_(
                    torch.empty(num_additional, embedding_dim),
                    mean=0.0, std=0.01
                ).to(self.device)

                pretrained_embeddings = torch.cat([pretrained_embeddings, additional_embeddings], dim=0)

            state_dict['mlp_i_embeddings.weight'] = pretrained_embeddings

        # Load the modified state dictionary
        self.load_state_dict(state_dict, strict=False)
        logging.info('Load model from ' + model_path)


    def count_variables(self) -> int:
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_parameters

    def actions_before_train(self):
        """
        Actions to perform before training starts.
        Can be used to re-initialize special parameters.
        """
        pass

    def actions_after_train(self):
        """
        Actions to perform after training completes.
        Can be used to save selected parameters or perform cleanup.
        """
        pass

    """
    Define Dataset Class
    """
    class Dataset(BaseDataset):
        def __init__(self, model, corpus, phase: str, train_set=None, dev_set=None, attribute=None):
            self.model = model  # model object reference
            self.corpus = corpus  # reader object reference
            self.phase = phase  # train / dev / test

            # if phase==test and test_all is true, then we load train_set and dev_set
            self.train_set = train_set
            self.dev_set = dev_set

            self.buffer_dict = dict()
            self.buffer = self.model.buffer and self.phase != 'train'
            self.topk = self.model.ndcg_topk
            self.reorg_train_data = self.model.reorg_train_data

            if self.phase == 'train' and self.reorg_train_data:
                global ItemAttributes
                ItemAttributes = attribute

                # Prepare and reorganize training data
                self.raw_data = corpus.data_df[phase]
                self.raw_data = self.raw_data[['user_id', 'item_id', 'rating']]
                self.raw_data = self.raw_data.groupby('user_id', as_index=False).agg({
                    'item_id': lambda x: list(x),
                    'rating': lambda x: list(x)
                })

                # Calculate number of positive items per user
                self.raw_data['pos_items'] = self.raw_data['item_id'].apply(len)

                # Calculate ideal DCG for each user
                logging.info(f"Computing IDCG with top-k: {self.topk}")
                self.raw_data['ideal_dcg'] = self.raw_data['rating'].apply(
                    lambda ratings: calculate_ideal_dcg(ratings, self.topk)
                )
                self.data = utils.df_to_dict(self.raw_data)
            else:
                self.raw_data = corpus.data_df[phase]
                self.data = utils.df_to_dict(self.raw_data)
            # else:
            #     self.data = utils.df_to_dict(corpus.data_df[phase])
            #     # ↑ DataFrame is not compatible with multi-thread operations

            if self.phase == 'test':
                self.max_train_pos_items = max(self.train_set.data['pos_items'])
                self.dev_pos_items = len(self.dev_set.data['item_id'][0])
                self.test_pos_items = len(self.data['item_id'][0])

                logging.info(f"Dataset statistics:")
                logging.info(f"  - Max training positive items: {self.max_train_pos_items}")
                logging.info(f"  - Dev positive items: {self.dev_pos_items}")
                logging.info(f"  - Test positive items: {self.test_pos_items}")

            self._prepare()

        def __len__(self):
            if type(self.data) == dict:
                for key in self.data:
                    return len(self.data[key])
            return len(self.data)

        def __getitem__(self, index: int) -> dict:
            return self.buffer_dict[index] if self.buffer else self._get_feed_dict(index)

        # Prepare model-specific variables and buffer feed dicts
        def _prepare(self) -> NoReturn:
            if self.buffer:
                for i in tqdm(range(len(self)), leave=False, desc=('Prepare ' + self.phase)):
                    self.buffer_dict[i] = self._get_feed_dict(i)

        def _get_feed_dict(self, index: int) -> dict:
            """
            Key method to construct input data for a single instance.
            Must be implemented by subclasses.

            Args:
                index (int): Index of the data instance

            Returns:
                dict: Feed dictionary for the instance
            """
            pass

        def actions_before_epoch(self) -> NoReturn:
            """
            Actions to perform before each training epoch.
            Can be used for data shuffling or other epoch-level preparations.
            """
            pass

        def collate_batch(self, feed_dicts: List[dict]) -> dict:
            """
            Collate a batch of feed dictionaries into a single batch dictionary.
            Handles variable-length sequences by padding.

            Args:
                feed_dicts (List[dict]): List of individual feed dictionaries

            Returns:
                dict: Batched feed dictionary with tensors
            """
            batch_dict = {}

            for key in feed_dicts[0]:
                if isinstance(feed_dicts[0][key], np.ndarray):
                    # Check if all arrays have the same length
                    lengths = [len(feed_dict[key]) for feed_dict in feed_dicts]
                    if any(length != lengths[0] for length in lengths):
                        # Variable length - use object array
                        stacked_values = np.array([feed_dict[key] for feed_dict in feed_dicts], dtype=object)
                    else:
                        # Fixed length - use regular array
                        stacked_values = np.array([feed_dict[key] for feed_dict in feed_dicts])
                else:
                    stacked_values = np.array([feed_dict[key] for feed_dict in feed_dicts])

                # Convert to tensor, padding if necessary
                if stacked_values.dtype == object:
                    # Pad sequences for variable-length data
                    batch_dict[key] = pad_sequence(
                        [torch.from_numpy(x) for x in stacked_values],
                        batch_first=True
                    )
                else:
                    batch_dict[key] = torch.from_numpy(stacked_values)

            batch_dict['batch_size'] = len(feed_dicts)
            batch_dict['phase'] = self.phase
            return batch_dict


class GeneralModel(BaseModel):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--num_neg', type=int, default=1,
                            help='The number of negative items during training.')
        parser.add_argument('--num_pos', type=int, default=1,
                            help='The number of positive items during training.')
        parser.add_argument('--dropout', type=float, default=0,
                            help='Dropout probability for each deep layer')
        parser.add_argument('--test_all', type=int, default=0,
                            help='Whether testing on all the items.')
        parser.add_argument('--loss_type', type=str, default='BPR',
                            choices=['RankNet', 'ListNet', 'ListMLE',
                                     'NeuralNDCG', 'ApproxNDCG', 'LambdaRank', 
                                     'Listwise_CE', 'NDCG', 'Listnet_Fair', 'BPR_Fair', 'Listmle_Fair'],          # ours
                            help='The loss used during training.')
        parser.add_argument('--psi_func', type=str, default='sigmoid',
                            choices=['hinge', 'softmax', 'sigmoid'],         
                            help='The psi_func used during training.')
        parser.add_argument('--fair_psi_func', type=str, default='sigmoid',
                            choices=['softmax', 'sigmoid'],          
                            help='The fair_psi_func used during training.')
        parser.add_argument('--neuralndcg_temp', type=float, default=1.0,
                            help='Temp for NeuralNDCG')
        parser.add_argument('--warmup_gamma', type=float, default=0.1,
                            help='Gamma for WARMUP-M.')
        parser.add_argument('--ndcg_gamma', type=float, default=0.1,
                            help='Gamma for NDCG-M.')
        parser.add_argument('--ndcg_topk', type=int, default=-1,
                            help='Topk for NDCG@k optimization')
        parser.add_argument('--fairness_c', type=float, default=0,
                            help='C for NDCG fairness loss')
        parser.add_argument('--balance_fair', action='store_true', default=True,
                            help='use balance fair for loss function')
        parser.add_argument('--e_mode', type=str, default='1',
                            help='mode for exy, default is 1, "avg" for average')
        parser.add_argument('--simple_fair', action='store_true', default=False,
                            help='use the simple fair loss (for ksong)')
        parser.add_argument('--tau_1', type=float, default=0.001,
                            help='tau_1)')
        parser.add_argument('--sigmoid_alpha', type=float, default=2.0,
                            help='sigmoid_alpha')
        parser.add_argument('--sigmoid_beta', type=float, default=2.0,
                            help='sigmoid_beta')
        parser.add_argument('--sigmoid_t', type=float, default=1.0,
                            help='sigmoid_t')
        parser.add_argument('--fair_type', type=str, default='exp_top1_fair',
                            help='fair_type from exp_topk, exp_top1_fair, exp_top1_fair_topk')
        parser.add_argument('--gamma2', type=float, default=0.5,
                            help='gamma2')
        parser.add_argument('--gamma3', type=float, default=0.5,
                            help='gamma3')
        parser.add_argument('--gamma4', type=float, default=0.5,
                            help='gamma4')

        return BaseModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.user_num = corpus.n_users
        self.item_num = corpus.n_items
        self.num_neg = args.num_neg
        self.num_pos = args.num_pos
        self.dropout = args.dropout
        self.test_all = args.test_all
        self.loss_type = args.loss_type
        self.neuralndcg_temp = args.neuralndcg_temp
        self.warmup_gamma = args.warmup_gamma
        self.ndcg_gamma = args.ndcg_gamma
        self.ndcg_topk = args.ndcg_topk
        self.fairness_c = args.fairness_c
        self.e_mode = args.e_mode
        self.eps = 1e-10
        self.balance_fair = args.balance_fair
        self.simple_fair = args.simple_fair
        self.tau_1 = args.tau_1
        self.sigmoid_alpha = args.sigmoid_alpha
        self.sigmoid_beta = args.sigmoid_beta
        self.sigmoid_t = args.sigmoid_t
        self.psi_func = args.psi_func
        self.fair_psi_func = args.fair_psi_func
        self.fair_type = args.fair_type
        self.gamma2 = args.gamma2
        self.gamma3 = args.gamma3
        self.gamma4 = args.gamma4
        super().__init__(args, corpus)
        self._build_loss_instance()

    def _build_loss_instance(self):
        if self.loss_type == 'Listwise_CE':
            self.warmup_loss = ndcg_loss.ListwiseCrossEntropyLoss(
                self.user_num, self.item_num, self.num_pos, self.warmup_gamma, self.eps,
                fairness_weight=self.fairness_c, expectation_mode=self.e_mode,
                use_balanced_fairness=self.balance_fair
            )
        elif self.loss_type == 'NDCG':
            self.ndcg_loss = ndcg_loss.NDCGLoss(
                self.user_num, self.item_num, self.num_pos, self.ndcg_gamma,
                top_k=self.ndcg_topk, fairness_weight=self.fairness_c,
                expectation_mode=self.e_mode, eps=self.eps,
                use_balanced_fairness=self.balance_fair, use_simple_fairness=self.simple_fair,
                threshold_tau1=self.tau_1, sigmoid_alpha=self.sigmoid_alpha,
                sigmoid_beta=self.sigmoid_beta, sigmoid_temperature=self.sigmoid_t,
                psi_function=self.psi_func, fair_psi_function=self.fair_psi_func,
                fairness_type=self.fair_type, gamma2=self.gamma2, gamma3=self.gamma3,
                gamma4=self.gamma4
            )
        elif self.loss_type == 'Listnet_Fair':
            self.listnet_fair_loss = ndcg_loss.ListNetFairLoss(
                self.num_pos, eps=self.eps, fairness_weight=self.fairness_c,
                expectation_mode=self.e_mode, use_balanced_fairness=self.balance_fair
            )
        elif self.loss_type == 'BPR_Fair':
            self.bpr_fair_loss = ndcg_loss.BPRFairLoss(
                eps=self.eps, fairness_weight=self.fairness_c,
                expectation_mode=self.e_mode, use_balanced_fairness=self.balance_fair
            )
        elif self.loss_type == 'Listmle_Fair':
            self.listmle_fair_loss = ndcg_loss.ListMLEFairLoss(
                self.num_pos, eps=self.eps, fairness_weight=self.fairness_c,
                expectation_mode=self.e_mode, use_balanced_fairness=self.balance_fair
            )

    def loss(self, out_dict: dict, epoch: int) -> torch.Tensor:
        """
        Calculate loss with multiple positive and negative samples.

        Args:
            out_dict (dict): Output dictionary containing predictions with shape [batch_size, num_pos + num_neg]
            epoch (int): Current training epoch

        Returns:
            torch.Tensor: Computed loss value
        """
        predictions = out_dict['prediction']  # [batch_size, num_pos + num_neg]
        batch_size = predictions.size(0)

        # Transform predictions for positive samples
        pos_predictions = torch.cat(
            torch.chunk(predictions[:, :self.num_pos], batch_size, dim=0), dim=1
        ).permute(1, 0)  # [batch_size * num_pos, 1]

        # Transform predictions for negative samples
        neg_predictions = torch.stack(
            [predictions[:, self.num_pos:]] * self.num_pos, dim=1
        ).view(-1, predictions.size(1) - self.num_pos)  # [batch_size * num_pos, num_neg]

        # Combine positive and negative predictions
        combined_predictions = torch.cat(
            [pos_predictions, neg_predictions], dim=1
        )  # [batch_size * num_pos, 1 + num_neg]

        pos_scores, neg_scores = combined_predictions[:, 0], combined_predictions[:, 1:]
        # [batch_size * num_pos], [batch_size * num_pos, num_neg]

        # Handle ratings dimension
        ratings = out_dict['rating']
        if len(ratings.shape) == 1:
            ratings = ratings[:, None]  # [batch_size, num_pos]
        
        # Calculate loss based on specified loss type
        if self.loss_type == 'RankNet':
            loss = losses.bpr_loss(pos_scores, neg_scores)
        elif self.loss_type == 'Listwise_CE':
            loss = self.warmup_loss(predictions, out_dict)
        elif self.loss_type == 'NDCG':
            loss = self.ndcg_loss(predictions, out_dict, epoch)
        elif self.loss_type == 'BPR_Fair':
            loss = self.bpr_fair_loss(pos_scores, neg_scores, predictions, out_dict)
        elif self.loss_type == 'Listmle_Fair':
            loss = self.listmle_fair_loss(predictions, out_dict)
        elif self.loss_type == 'Listnet_Fair':
            loss = self.listnet_fair_loss(predictions, out_dict)
        elif self.loss_type == 'NeuralNDCG':
            loss = losses.neural_sort_loss(
                predictions, ratings, self.device, temperature=self.neuralndcg_temp
            )
        elif self.loss_type == 'ApproxNDCG':
            loss = losses.approx_ndcg_loss(predictions, ratings, self.device)
        elif self.loss_type == 'ListNet':
            loss = losses.listnet_loss(predictions, ratings, self.device)
        elif self.loss_type == 'ListMLE':
            loss = losses.listmle_loss(predictions, ratings, self.device)
        elif self.loss_type == 'LambdaRank':
            loss = losses.lambda_loss(predictions, ratings, self.device, 'lambdaRank_scheme')
        else:
            raise NotImplementedError(f"Loss type '{self.loss_type}' is not implemented")

        return loss

    class Dataset(BaseModel.Dataset):
        def _get_feed_dict(self, index):
            if self.phase == 'train':
                if not self.reorg_train_data:
                    user_id, target_item = self.data['user_id'][index], self.data['item_id'][index]
                    num_pos_items = self.data['pos_items'][index]
                    ideal_dcg = self.data['ideal_dcg'][index]
                    clicked_list = self.corpus.train_clicked_list[user_id]
                    ratings_list = self.corpus.train_ratings_list[user_id]
                    idx = np.random.choice(np.arange(len(clicked_list)), self.model.num_pos, replace=False)
                    pos_items, pos_ratings = clicked_list[idx], ratings_list[idx]
                    neg_items = np.random.randint(1, self.corpus.n_items, size=(self.model.num_neg))
                    for j in range(self.model.num_neg):
                        while neg_items[j] in clicked_list:
                            neg_items[j] = np.random.randint(1, self.corpus.n_items)            
                    if self.model.num_pos == 1:
                        item_ids = np.concatenate([[target_item], neg_items]).astype(int)
                        rating = self.data['rating'][index]
                    elif self.model.num_pos > 1:
                        item_ids = np.concatenate([pos_items, neg_items]).astype(int)
                        rating = pos_ratings
                    else:
                        assert 0, "num_pos must be >= 1"
                    fairness_attrs = self._generate_fairness_attributes(item_ids)
                    feed_dict = {
                        'user_id': user_id,
                        'item_id': item_ids,
                        'rating': rating,
                        'num_pos_items': num_pos_items,
                        'ideal_dcg': ideal_dcg,
                        'a_index': fairness_attrs['a_index'],
                        'b_index': fairness_attrs['b_index'],
                        'mask_ratio_a': fairness_attrs['mask_ratio_a'],
                        'mask_ratio_b': fairness_attrs['mask_ratio_b'],
                        'rho': fairness_attrs['rho']
                    }
                else:
                    user_id = self.data['user_id'][index]
                    num_pos_items = self.data['pos_items'][index]
                    ideal_dcg = self.data['ideal_dcg'][index]
                    clicked_list = np.array(self.data['item_id'][index])
                    ratings_list = np.array(self.data['rating'][index])
                    idx = np.random.choice(np.arange(len(clicked_list)), self.model.num_pos, replace=False)
                    pos_items, pos_ratings = clicked_list[idx], ratings_list[idx]
                    if self.model.num_neg <= 2000:
                        neg_items = np.random.randint(1, self.corpus.n_items, size=(self.model.num_neg))
                        for j in range(self.model.num_neg):
                            while neg_items[j] in clicked_list:
                                neg_items[j] = np.random.randint(1, self.corpus.n_items)
                    else:
                        neg_items_cand = np.setdiff1d(np.arange(1, self.corpus.n_items), clicked_list, assume_unique=True)
                        replace = False if len(neg_items_cand)>= self.model.num_neg else True
                        neg_items = np.random.choice(neg_items_cand, self.model.num_neg, replace=replace)
                    if self.model.num_pos > 1:
                        item_ids = np.concatenate([pos_items, neg_items]).astype(int)
                    elif self.model.num_pos == 1:
                        item_ids = np.concatenate([pos_items, neg_items]).astype(int)
                    else:
                        assert 0, "num_pos must be >= 1"
                    fairness_attrs = self._generate_fairness_attributes(item_ids)
                    feed_dict = {
                        'user_id': user_id,
                        'item_id': item_ids,
                        'rating': pos_ratings,
                        'num_pos_items': num_pos_items,
                        'ideal_dcg': ideal_dcg,
                        'a_index': fairness_attrs['a_index'],
                        'b_index': fairness_attrs['b_index'],
                        'mask_ratio_a': fairness_attrs['mask_ratio_a'],
                        'mask_ratio_b': fairness_attrs['mask_ratio_b'],
                        'rho': fairness_attrs['rho']
                    }
                return feed_dict
            else:    # self.phase == 'dev' or 'test'
                user_id, target_items, rating = self.data['user_id'][index], self.data['item_id'][index], self.data['rating'][index]
                if self.model.test_all and self.phase == 'test':
                    # For test_all mode, exclude items from train, dev, and test sets
                    excluded_items = np.concatenate([
                        target_items,
                        self.train_set.data['item_id'][index],
                        self.dev_set.data['item_id'][index]
                    ])
                    neg_items = np.setdiff1d(np.arange(1, self.corpus.n_items), excluded_items)
                    # Sample with replacement to ensure consistent number of items across users
                    neg_items = np.random.choice(neg_items, self.corpus.n_items - 1, replace=True)
                else:
                    neg_items = self.data['neg_items'][index]
                item_ids = np.concatenate([target_items, neg_items]).astype(int)
                fairness_attrs = self._generate_fairness_attributes(item_ids)
                feed_dict = {
                    'user_id': user_id,
                    'item_id': item_ids,
                    'rating': rating,
                    'a_index': fairness_attrs['a_index'],
                    'b_index': fairness_attrs['b_index'],
                    'mask_ratio_a': fairness_attrs['mask_ratio_a'],
                    'mask_ratio_b': fairness_attrs['mask_ratio_b'],
                    'rho': fairness_attrs['rho']
                }
                return feed_dict

        def _generate_fairness_attributes(self, item_ids):
            """
            Generate fairness-related attributes for a set of items.

            Args:
                item_ids (np.ndarray): Array of item IDs

            Returns:
                dict: Dictionary containing fairness attributes including rho, indices, and mask ratios
            """
            fairness_info = {
                'rho': [],
                'a_index': [],
                'b_index': [],
                'mask_ratio_a': [],
                'mask_ratio_b': []
            }

            global ItemAttributes
            num_items = len(item_ids)

            # Create binary masks for sensitive attribute groups
            group_a_mask = np.array([
                1 if is_sensitive else 0
                for is_sensitive in ItemAttributes['sensitive'][item_ids - 1]
            ])
            group_b_mask = np.where(group_a_mask == 1, 0, 1)

            count_a = np.sum(group_a_mask)
            count_b = np.sum(group_b_mask)

            # Calculate rho (ratio of minority to majority group)
            if count_a < count_b:
                fairness_info['rho'] = count_a / count_b if count_b > 0 else 0.0
            else:
                fairness_info['rho'] = count_b / count_a if count_a > 0 else 0.0
                # Swap groups so that group A is always the minority
                group_a_mask, group_b_mask = group_b_mask, group_a_mask
                count_a, count_b = count_b, count_a

            # Calculate mask ratios (avoid division by zero)
            fairness_info['mask_ratio_a'] = count_a / num_items if num_items > 0 else 0.0
            fairness_info['mask_ratio_b'] = count_b / num_items if num_items > 0 else 0.0

            # Ensure mask_ratio_a is not zero to avoid division by zero in loss calculations
            if fairness_info['mask_ratio_a'] == 0:
                fairness_info['mask_ratio_a'] = 1.0

            fairness_info['a_index'] = group_a_mask
            fairness_info['b_index'] = group_b_mask

            return fairness_info

        def actions_before_epoch(self) -> NoReturn:
            """
            Actions to perform before each training epoch.
            Note: Positive and negative sampling is now handled in _get_feed_dict method.
            """
            pass


class SequentialModel(GeneralModel):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--history_max', type=int, default=20,
                            help='Maximum length of history.')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.history_max = args.history_max
        super().__init__(args, corpus)

    class Dataset(GeneralModel.Dataset):
        def _prepare(self):
            idx_select = np.array(self.data['position']) > 0  # history length must be non-zero
            for key in self.data:
                self.data[key] = np.array(self.data[key])[idx_select]
            super()._prepare()

        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)
            pos = self.data['position'][index]
            user_seq = self.corpus.user_his[feed_dict['user_id']][:pos]
            if self.model.history_max > 0:
                user_seq = user_seq[-self.model.history_max:]
            feed_dict['history_items'] = np.array([x[0] for x in user_seq])
            feed_dict['history_times'] = np.array([x[1] for x in user_seq])
            feed_dict['lengths'] = len(feed_dict['history_items'])
            return feed_dict
