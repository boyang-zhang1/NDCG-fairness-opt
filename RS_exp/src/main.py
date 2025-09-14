# -*- coding: UTF-8 -*-
# source ~/.bashrc ; conda activate ReChorus; cd /data/qiuzh/NDCG_MAP_Opt/src

import os
import sys
import pickle
import logging
import argparse
import numpy as np
import torch
import time
import pandas as pd
from helpers import *
from models.general import *
from models.sequential import *
from models.developing import *
from utils import utils


def parse_global_args(parser):
    parser.add_argument('--gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES')
    parser.add_argument('--verbose', type=int, default=logging.INFO,
                        help='Logging Level, 0, 10, ..., 50')
    parser.add_argument('--log_file', type=str, default='',
                        help='Logging file path')
    parser.add_argument('--random_seed', type=int, default=0,
                        help='Random seed of numpy and pytorch.')
    parser.add_argument('--load', type=int, default=0,
                        help='Whether load model and continue to train')
    parser.add_argument('--pretrain_model', type=str, default='',
                        help='path to pretrain model')
    parser.add_argument('--init_last', type=int, default=0,
                        help='Whether to initialize the last layer')
    parser.add_argument('--fix_embedding', type=int, default=0,
                        help='Whether to fix fix embedding layer')
    parser.add_argument('--train', type=int, default=1,
                        help='To train the model or not.')
    parser.add_argument('--regenerate', type=int, default=0,
                        help='Whether to regenerate intermediate files.')
    parser.add_argument('--run_name', type=str, default='not specified',
                        help='Notes of the current experiment')

    return parser


def main():
    logging.info('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)
    exclude = ['check_epoch', 'path', 'pin_memory', 'regenerate', 'sep', 'train', 
               'verbose', 'metric', 'test_epoch', 'buffer', 'model_path', 'log_file']
    logging.info(utils.format_arg_str(args, exclude_lst=exclude))

    # Random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True

    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logging.info('GPU available: {}'.format(torch.cuda.is_available()))

    # Read data
    corpus_path = os.path.join(args.path, args.dataset, model_name.reader + '.pkl')
    if not args.regenerate and os.path.exists(corpus_path):
        logging.info('Load corpus from {}'.format(corpus_path))
        corpus = pd.read_pickle(corpus_path)
    else:
        corpus = reader_name(args)
        logging.info('Save corpus to {}'.format(corpus_path))
        corpus.to_pickle(corpus_path)
    attribute = corpus.data_df['attribute']

    # Define model
    model = model_name(args, corpus)
    logging.info(model)
    model.apply(model.init_weights)
    model.actions_before_train()
    model.to(model.device)

    # Run model
    data_dict = dict()
    # for phase in ['train', 'dev']:
    #     data_dict[phase] = model_name.Dataset(model, corpus, phase)
    data_dict['train'] = model_name.Dataset(model, corpus, 'train', attribute=attribute)
    data_dict['dev'] = model_name.Dataset(model, corpus, 'dev')
    data_dict['test'] = model_name.Dataset(model, corpus, 'test', data_dict['train'], data_dict['dev'])
    runner = runner_name(args)
    # logging.info('Test Before Training: ' + runner.print_res(data_dict['test']))
    if args.load > 0:
        model.load_model(model_path=args.pretrain_model)
        
        if args.train == 0 and args.test_all > 0:
            logging.info(os.linesep + 'Test After Loading: ' + runner.print_res(data_dict['test']))
            return

        logging.info(os.linesep + 'Dev After Loading: ' + runner.print_res(data_dict['dev'], attribute))
        logging.info(os.linesep + 'Test After Loading: ' + runner.print_res(data_dict['test'], attribute))
        if args.init_last > 0:
            logging.info('Reinitialize the last layer.')
            model.reset_last_layer()
        if args.fix_embedding > 0:
            logging.info('Fix embedding layer.')
            model.fix_embedding_layer()
    if args.train > 0:
        runner.train(data_dict, attribute)
    # logging.info(os.linesep + 'Test After Training: ' + runner.print_res(data_dict['test'], attribute))

    model.actions_after_train()
    logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)


if __name__ == '__main__':
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--model_name', type=str, default='BPR', help='Choose a model to run.')
    init_args, init_extras = init_parser.parse_known_args()
    model_name = eval('{0}.{0}'.format(init_args.model_name))
    reader_name = eval('{0}.{0}'.format(model_name.reader))
    runner_name = eval('{0}.{0}'.format(model_name.runner))

    # Args
    parser = argparse.ArgumentParser(description='')
    parser = parse_global_args(parser)
    parser = reader_name.parse_data_args(parser)
    parser = runner_name.parse_runner_args(parser)
    parser = model_name.parse_model_args(parser)
    args, extras = parser.parse_known_args()

    # Logging configuration
    log_args = [init_args.model_name, args.dataset, str(args.random_seed), args.run_name]
    for arg in ['ndcg_topk', 'fairness_c'] + model_name.extra_log_args:
        log_args.append(arg + '=' + str(eval('args.' + arg)))
    log_file_name = '__'.join(log_args).replace(' ', '__')
    current_time = time.asctime(time.localtime(time.time())).replace(' ', '-')
    log_file_name += '__' + current_time
    if args.log_file == '':
        args.log_file = '../log/{}/{}.txt'.format(init_args.model_name, log_file_name)
    else:
        args.log_file = os.path.join(args.log_file, '{}/{}.txt'.format(init_args.model_name, log_file_name))
    if args.model_path == '':
        args.model_path = '../model/{}/{}.pt'.format(init_args.model_name, log_file_name)
    else:
        args.model_path = os.path.join(args.model_path, '{}/{}.pt'.format(init_args.model_name, log_file_name))

    utils.check_dir(args.log_file)
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(init_args)

    main()
