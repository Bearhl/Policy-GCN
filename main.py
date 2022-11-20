# -*- coding: UTF-8 -*-

import torch
import random
import argparse
import numpy as np
import fitlog
import six
from graph_model import Policy_GCN
from Policy import Policy_
from data_process import DataProcess
import warnings


class instruction:
    def __init__(self, opt):
        self.opt = opt
        self.Dataproc = DataProcess(self.opt)
        self.opt.embedding_matrix = self.Dataproc.embedding_matrix
        self.all_data = self.Dataproc.all_data

        _, _, self.train_loader, self.test_loader, self.keys = self.load_data()

        self.GCN = Policy_GCN(self.opt).to(self.opt.device)
        self.Policy = Policy_(self.opt)

    def load_data(self):

        train_data = self.all_data['train']
        test_data = self.all_data['test']
        train_dataloader, test_dataloader = self.Dataproc.build_dataloader()

        return train_data, test_data, train_dataloader, test_dataloader, train_data[0].keys()

    def train(self):
        # 对GCN模型采用延迟的方法进行warmup权重更新
        self.GCN.warmup_train([self.train_loader, self.test_loader])
        # 对Policy网络和GCN网络进行训练
        self.Policy.run([self.train_loader, self.test_loader], self.GCN.targetnet)
        # 对GCN网络进行训练
        # self.GCN.train_gcn(self.Policy.actor_target, [self.train_loader, self.test_loader])

    def run(self):
        fitlog.set_log_dir('./logs/')
        for arg, value in sorted(six.iteritems(vars(self.opt))):
            fitlog.add_hyper({arg: value})  # 通过这种方式记录ArgumentParser的参数

        self.train()

        fitlog.finish()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser("Policy-GNN for ABSA")

    parser.add_argument('--dataset', type=str, default='rest14',
                        help='choose the dataset from twitter/rest14/lap14/rest15/rest16')
    parser.add_argument('--seed', type=int, default=random.randint(0, 100))


    parser.add_argument('--batchsize', type=int, default=64, help='batchsize for training or testing')
    parser.add_argument('--hidden_dim', type=int, default=300, help='hidden for lstm and GCN')
    parser.add_argument('--embed_dim', type=int, default=300, help='embedding for lstm and GCN')
    # parser.add_argument('--max_length', type=int, default=80, help='max length for each sentences')
    parser.add_argument('--polarity', type=int, default=3, help='the number of polarity classes')
    parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--warmup_epoch', type=int, default=20)  # 30
    parser.add_argument('--max_jump', type=int, default=4)
    parser.add_argument('--epoch', type=int, default=1)  # 30

    parser.add_argument('--max_episodes', type=int, default=20, help='the episodes for dqn training')  # 每个episode需要4分钟
    # parser.add_argument('--memory_size', type=int, default=500, help='the memory size for save dqn memory')
    parser.add_argument('--k', type=int, default=3, help='the number of action predict')
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.1)
    parser.add_argument('--epsilon_decay_steps', type=int, default=100)
    parser.add_argument('--update_steps', type=int, default=100)
    parser.add_argument('--samplecnt', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--debug', type=bool, default=True)

    opt = parser.parse_args()

    opt.device = "cpu" if not torch.cuda.is_available() else f"cuda:{opt.device}"

    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ins = instruction(opt)

    ins.run()
