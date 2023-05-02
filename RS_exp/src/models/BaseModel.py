# -*- coding: UTF-8 -*-

import time
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


def cal_ideal_dcg(x, topk=-1):
    x_sorted = -np.sort(-np.array(x))  # descending order
    pos = np.log2(1.0 + np.arange(1, len(x)+1))
    ideal_dcg_size = topk if topk != -1 else len(x)
    ideal_dcg = np.sum(((2 ** x_sorted - 1) / pos)[:ideal_dcg_size])
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
        self.check_list = list()  # observe tensors in check_list every check_epoch

        self._define_params()
        self.total_parameters = self.count_variables()
        logging.info('#params: %d' % self.total_parameters)

    """
    Key Methods
    """
    def _define_params(self) -> NoReturn:
        pass

    def forward(self, feed_dict: dict) -> dict:
        """
        :param feed_dict: batch prepared in Dataset
        :return: out_dict, including prediction with shape [batch_size, n_candidates]
        """
        pass

    def loss(self, out_dict: dict) -> torch.Tensor:
        pass

    """
    Auxiliary Methods
    """
    def customize_parameters(self) -> list:
        # customize optimizer settings for different parameters
        weight_p, bias_p = [], []
        for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        optimize_dict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}]
        return optimize_dict

    def save_model(self, model_path=None) -> NoReturn:
        if model_path is None:
            model_path = self.model_path
        utils.check_dir(model_path)
        torch.save(self.state_dict(), model_path)

    def load_model(self, model_path=None) -> NoReturn:
        if model_path is None:
            model_path = self.model_path
        self.load_state_dict(torch.load(model_path, map_location=self.device))
        logging.info('Load model from ' + model_path)

    def count_variables(self) -> int:
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_parameters

    def actions_before_train(self):  # e.g., re-initial some special parameters
        pass

    def actions_after_train(self):  # e.g., save selected parameters
        pass

    """
    Define Dataset Class
    """
    class Dataset(BaseDataset):
        def __init__(self, model, corpus, phase: str, train_set=None, dev_set=None):
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
                self.raw_data = corpus.data_df[phase]
                self.raw_data = self.raw_data[['user_id', 'item_id', 'rating']]
                self.raw_data = self.raw_data.groupby('user_id', as_index=False).agg({'item_id':lambda x: list(x), 'rating':lambda x: list(x)})
                self.raw_data['pos_items'] = self.raw_data['item_id'].apply(lambda x: len(x))
                print("current topk:", self.topk)
                self.raw_data['ideal_dcg'] = self.raw_data['rating'].apply(lambda x: cal_ideal_dcg(x, self.topk))
                self.data = utils.df_to_dict(self.raw_data)
            else:
                self.data = utils.df_to_dict(corpus.data_df[phase])
                # ↑ DataFrame is not compatible with multi-thread operations

            if self.phase == 'test':
                self.max_train_pos_items = max(self.train_set.data['pos_items'])
                self.dev_pos_items = len(self.dev_set.data['item_id'][0])
                self.test_pos_items = len(self.data['item_id'][0])
                print("train_max_pos_items:", self.max_train_pos_items)
                print("dev_pos_items:", self.dev_pos_items)
                print("test_pos_items:", self.test_pos_items)

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

        # ! Key method to construct input data for a single instance
        def _get_feed_dict(self, index: int) -> dict:
            pass

        # Called before each training epoch
        def actions_before_epoch(self) -> NoReturn:
            pass

        # Collate a batch according to the list of feed dicts
        def collate_batch(self, feed_dicts: List[dict]) -> dict:
            feed_dict = dict()
            for key in feed_dicts[0]:
                if isinstance(feed_dicts[0][key], np.ndarray):
                    tmp_list = [len(d[key]) for d in feed_dicts]
                    if any([tmp_list[0] != l for l in tmp_list]):
                        stack_val = np.array([d[key] for d in feed_dicts], dtype=np.object)
                    else:
                        stack_val = np.array([d[key] for d in feed_dicts])
                else:
                    stack_val = np.array([d[key] for d in feed_dicts])
                if stack_val.dtype == np.object:  # inconsistent length (e.g., history)
                    feed_dict[key] = pad_sequence([torch.from_numpy(x) for x in stack_val], batch_first=True)
                else:
                    feed_dict[key] = torch.from_numpy(stack_val)
            feed_dict['batch_size'] = len(feed_dicts)
            feed_dict['phase'] = self.phase
            return feed_dict


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
                                     'Listwise_CE', 'NDCG'],          # ours
                            help='The loss used during training.')
        parser.add_argument('--neuralndcg_temp', type=float, default=1.0,
                            help='Temp for NeuralNDCG')
        parser.add_argument('--warmup_gamma', type=float, default=0.1,
                            help='Gamma for WARMUP-M.')
        parser.add_argument('--ndcg_gamma', type=float, default=0.1,
                            help='Gamma for NDCG-M.')
        parser.add_argument('--ndcg_topk', type=int, default=-1,
                            help='Topk for NDCG@k optimization')
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
        self.eps = 1e-10
        super().__init__(args, corpus)
        self._build_loss_instance()

    def _build_loss_instance(self):
        if self.loss_type == 'Listwise_CE':
            self.warmup_loss = ndcg_loss.Listwise_CE_Loss(self.user_num, self.item_num, self.num_pos, self.warmup_gamma, self.eps)
        elif self.loss_type == 'NDCG':
            self.NDCG_loss = ndcg_loss.NDCG_Loss(self.user_num, self.item_num, self.num_pos, self.ndcg_gamma, k=self.ndcg_topk)

    def loss(self, out_dict: dict, epoch: int) -> torch.Tensor:
        """
        has multiple postive and nagetive samples

        :param out_dict: contain prediction with [batch_size, -1], the first column for positive, the rest for negative
        :return:
        """
        predictions = out_dict['prediction']                                                                 # [batch_size, num_pos + num_neg]
        batch_size = predictions.size(0)
        pos_preds_transformed = torch.cat(torch.chunk(predictions[:, :self.num_pos], batch_size, dim=0), dim=1).permute(1,0)  # [batch_size * num_pos, 1]
        neg_preds_transformed = torch.repeat_interleave(predictions[:, self.num_pos:], self.num_pos, dim=0)                   # [batch_size * num_pos, num_neg]
        preds_transformed = torch.cat([pos_preds_transformed, neg_preds_transformed], dim=1)     # [batch_size * num_pos, 1+num_neg]
        pos_pred, neg_pred = preds_transformed[:, 0], preds_transformed[:, 1:]                   # [batch_size * num_pos], [batch_size * num_pos, num_neg]
        ratings = out_dict['rating'] if len(out_dict['rating'].shape)==2 else out_dict['rating'][:, None]    # [batch_size, num_pos]
        
        if self.loss_type == 'RankNet':
            loss = losses.bpr_loss(pos_pred, neg_pred)
        elif self.loss_type == 'Listwise_CE':
            loss = self.warmup_loss(predictions, out_dict)
        elif self.loss_type == 'NDCG':
            loss = self.NDCG_loss(predictions, out_dict)
        elif self.loss_type == 'NeuralNDCG':
            loss = losses.neural_sort_loss(predictions, ratings, self.device, temperature=self.neuralndcg_temp)
        elif self.loss_type == 'ApproxNDCG':
            loss = losses.approx_ndcg_loss(predictions, ratings, self.device)
        elif self.loss_type == 'ListNet':
            loss = losses.listnet_loss(predictions, ratings, self.device)
        elif self.loss_type == 'ListMLE':
            loss = losses.listmle_loss(predictions, ratings, self.device)
        elif self.loss_type == 'LambdaRank':
            loss = losses.lambda_loss(predictions, ratings, self.device, 'lambdaRank_scheme')
        else:
            raise NotImplementedError
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
                    feed_dict = {
                        'user_id': user_id,
                        'item_id': item_ids,
                        'rating': rating,
                        'num_pos_items': num_pos_items,
                        'ideal_dcg': ideal_dcg
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
                    feed_dict = {
                        'user_id': user_id,
                        'item_id': item_ids,
                        'rating': pos_ratings,
                        'num_pos_items': num_pos_items,
                        'ideal_dcg': ideal_dcg
                    }
                return feed_dict
            else:    # self.phase == 'dev' or 'test'
                user_id, target_items, rating = self.data['user_id'][index], self.data['item_id'][index], self.data['rating'][index]
                if self.model.test_all and self.phase=='test':
                    neg_items = np.setdiff1d(np.arange(1, self.corpus.n_items), target_items)
                    neg_items = np.setdiff1d(neg_items, self.train_set.data['item_id'][index])
                    neg_items = np.setdiff1d(neg_items, self.dev_set.data['item_id'][index])
                    neg_items = np.random.choice(neg_items, self.corpus.n_items-1, replace=True)
                    # make all users have the same number of negative items
                else:
                    neg_items = self.data['neg_items'][index]
                item_ids = np.concatenate([target_items, neg_items]).astype(int)
                feed_dict = {
                    'user_id': user_id,
                    'item_id': item_ids,
                    'rating': rating
                }
                return feed_dict

        # Sample positive and negative items for all the instances
        # use this function ONLY before training
        # NOW! we put positve and negative sampling into _get_feed_dict !!!
        def actions_before_epoch(self) -> NoReturn:
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
