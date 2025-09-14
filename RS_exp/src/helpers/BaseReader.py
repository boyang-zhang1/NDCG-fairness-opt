# -*- coding: UTF-8 -*-

import os
import pickle
import argparse
import logging
import numpy as np
import pandas as pd
from typing import NoReturn

from utils import utils


class BaseReader(object):
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--path', type=str, default='../data/',
                            help='Input data dir.')
        parser.add_argument('--dataset', type=str, default='ml-20m',
                            help='Choose a dataset.')
        parser.add_argument('--sep', type=str, default='\t',
                            help='sep of csv file.')
        parser.add_argument('--sensitive_types', type=str, default='Action,Crime',
                            help='The sensitive types for attribute, sep by ",", e.g. Action,Crime for movies and F for gender.')
        parser.add_argument('--attribute', type=str, default='users',
                            help='users/movies has attributes(read users/movies.dat)')
        
        return parser

    def __init__(self, args):
        self.sep = args.sep
        self.prefix = args.path
        self.dataset = args.dataset
        self.sensitive_types = args.sensitive_types.split(',')
        self.attribute =args.attribute

        self._read_data()
        self._append_his_info()

    def _read_data(self) -> NoReturn:
        logging.info('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
        self.data_df = dict()

        self.data_df['attribute'] = pd.read_csv(os.path.join(self.prefix, self.dataset, self.attribute + '.dat'), sep=self.sep, encoding='latin-1')
        self.data_df['attribute']['sensitive'] = False
        if self.attribute == 'movies':
            for i in range(len(self.data_df['attribute']['genres'])):
                self.data_df['attribute']['genres'][i] = self.data_df['attribute']['genres'][i].split('|')
                for genre in self.data_df['attribute']['genres'][i]:
                    if genre in self.sensitive_types:
                        self.data_df['attribute'].loc[i, 'sensitive'] = True
        elif self.attribute == 'users':
            for i in range(len(self.data_df['attribute']['gender'])):
                if self.data_df['attribute']['gender'][i] == self.sensitive_types[0]:
                    self.data_df['attribute']['sensitive'][i] = True
                    
        elif self.attribute == 'age':
            for i in range(len(self.data_df['attribute']['age'])):
                if str(self.data_df['attribute']['age'][i]) in self.sensitive_types:
                    self.data_df['attribute'].loc[i, 'sensitive'] = True

        elif self.attribute == 'occupation':
            for i in range(len(self.data_df['attribute']['occupation'])):
                if str(self.data_df['attribute']['occupation'][i]) in self.sensitive_types:
                    self.data_df['attribute'].loc[i, 'sensitive'] = True

        elif self.attribute == 'books':
            for i in range(len(self.data_df['attribute']['genres'])):
                if self.data_df['attribute']['genres'][i] == self.sensitive_types[0]:
                    self.data_df['attribute'].loc[i, 'sensitive'] = True

        elif self.attribute == 'movies_year':
            for i in range(len(self.data_df['attribute']['year'])):
                if int(self.data_df['attribute']['year'][i]) < int(self.sensitive_types[0]):
                    self.data_df['attribute'].loc[i, 'sensitive'] = True
        print("Load attribute finish")

        for key in ['train', 'dev', 'test']:
            self.data_df[key] = pd.read_csv(os.path.join(self.prefix, self.dataset, key + '.csv'), sep=self.sep)
            self.data_df[key] = utils.eval_list_columns(self.data_df[key])
        print("Read csv finish")

        logging.info('Counting dataset statistics...')

        self.dev_test_df = pd.concat([df[['user_id','item_id']] for df in [self.data_df['dev'], self.data_df['test']]])

        self.n_users = max(self.data_df['train']['user_id'].max(), self.dev_test_df['user_id'].max()) + 1
        
        self.n_items = max(self.data_df['train']['item_id'].max(), self.dev_test_df['item_id'].apply(max).max(), np.array(self.data_df['dev']['neg_items'].tolist()).max(), np.array(self.data_df['test']['neg_items'].tolist()).max()) + 1
        self.n_entry = len(self.data_df['train']) + \
                        (self.n_users-1) * self.data_df['dev']['item_id'].apply(lambda x: len(x)).max() + \
                        (self.n_users-1) * self.data_df['test']['item_id'].apply(lambda x: len(x)).max()

        for key in ['dev', 'test']:
            if 'neg_items' in self.data_df[key]:
                neg_items = np.array(self.data_df[key]['neg_items'].tolist())
                assert (neg_items >= self.n_items).sum() == 0  # assert negative items don't include unseen ones

        logging.info('"# user": {}, "# item": {}, "# entry": {}'.format(
            self.n_users - 1, self.n_items - 1, self.n_entry))

    def _append_his_info(self) -> NoReturn:
        """
        Add history info to data_df: position
        ! Need data_df to be sorted by time in ascending order
        disable time here!
        """
        logging.info('Appending history info...')
        self.user_his = dict()           # store the already seen sequence of each user
        self.train_clicked_set = dict()  # store the clicked item set of each user in training set
        self.train_clicked_list = dict()
        self.train_ratings_list = dict()
        # process train_df first
        df = self.data_df['train']
        position = list()
        for uid, iid, rt in zip(df['user_id'], df['item_id'], df['rating']):
            if uid not in self.user_his:
                self.user_his[uid] = list()
                self.train_clicked_set[uid] = set()
                self.train_clicked_list[uid] = list() 
                self.train_ratings_list[uid] = list()
            position.append(len(self.user_his[uid]))
            self.user_his[uid].append((iid, -1))    # disable time here
            self.train_clicked_set[uid].add(iid)
            self.train_clicked_list[uid].append(iid)
            self.train_ratings_list[uid].append(rt)
        df['position'] = position
        for k, v in self.train_clicked_list.items():
            self.train_clicked_list[k] = np.array(v)
        for k, v in self.train_ratings_list.items():
            self.train_ratings_list[k] = np.array(v)
        # process dev_df and test_df next
        for key in ['dev', 'test']:
            df = self.data_df[key]
            position = list()
            for uid, iids in zip(df['user_id'], df['item_id']):
                position.append(len(self.user_his[uid]))
            df['position'] = position


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser = BaseReader.parse_data_args(parser)
    args, extras = parser.parse_known_args()

    args.path = '../../data/'
    corpus = BaseReader(args)

    corpus_path = os.path.join(args.path, args.dataset, 'BaseReader.pkl')
    logging.info('Save corpus to {}'.format(corpus_path))
    pickle.dump(corpus, open(corpus_path, 'wb'))
