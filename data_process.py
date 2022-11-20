import os
import math

import spacy
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
from dependency_graph import WhitespaceTokenizer


def load_word_vec(path, word2idx=None, embed_dim=300):
    # 此段函数根据word2idx中的单词，找到对应的词嵌入并返回（此时顺序可能跟word2idx的不一样）
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dataset):
    # 此段函数将返回word2idx一一对应的词嵌入
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), dataset)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors ...')
        # 初始化全为0，第一行是一个正态分布的向量
        embedding_matrix = np.zeros((len(word2idx), embed_dim))  # idx 0 and 1 are all-zeros
        # 前面两个参数应该是均值标准差、最后一个参数是维度
        embedding_matrix[1, :] = np.random.uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim), (1, embed_dim))
        # 加载存在单词表中的词嵌入
        fname = './glove/glove.840B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        print('building embedding_matrix:', embedding_matrix_file_name)
        # 填充/或者理解为将上面乱序的word_vec排列好顺序
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix


class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            # 字典中前两个分别是 <pad> 和<unk>
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v: k for k, v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
        text = text.lower()
        words = text.split()
        unknownidx = 1
        # 将text字符转换成idx 组成squence
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return sequence


class DataProcess:
    def __init__(self, opt, embedding_dim=300):
        self.file_name = {
            'twitter': {
                'train': './datasets/acl-14-short-data/train.raw',
                'test': './datasets/acl-14-short-data/test.raw'
            },
            'rest14': {
                'train': './datasets/semeval14/restaurant_train.raw',
                'test': './datasets/semeval14/restaurant_test.raw'
            },
            'lap14': {
                'train': './datasets/semeval14/laptop_train.raw',
                'test': './datasets/semeval14/laptop_test.raw'
            },
            'rest15': {
                'train': './datasets/semeval15/restaurant_train.raw',
                'test': './datasets/semeval15/restaurant_test.raw'
            },
            'rest16': {
                'train': './datasets/semeval16/restaurant_train.raw',
                'test': './datasets/semeval16/restaurant_test.raw'
            }
        }
        self.opt = opt
        self.opt.tokenizer = spacy.load('en_core_web_sm')
        self.opt.tokenizer.tokenizer = WhitespaceTokenizer(self.opt.tokenizer.vocab)
        self.dataset = self.opt.dataset
        # self.embedding_matrix = self.opt.embedding_matrix
        self.embedding_dim = embedding_dim

        self.all_data = dict()
        self.text = ''
        self.word2idx = None
        self.opt.max_len = 30

        self._init()

    def load_text(self, dataset):
        for i in ['train', 'test']:
            file = self.file_name[dataset][i]
            with open(file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for t in range(0, len(lines), 3):
                    self.text += lines[t].replace('$T$', lines[t+1].strip()).strip() + ' '

    def load_dict(self):
        if os.path.exists(self.dataset + '_word2idx.pkl'):
            print("loading {0} tokenizer...".format(self.dataset))
            with open(self.dataset+'_word2idx.pkl', 'rb') as f:
                self.word2idx = pickle.load(f)
                self.tokenizer = Tokenizer(word2idx=self.word2idx)
        else:
            self.tokenizer = Tokenizer()
            self.tokenizer.fit_on_text(self.text)
            with open(self.dataset + '_word2idx.pkl', 'wb') as f:
                pickle.dump(self.tokenizer.word2idx, f)

    def load_data(self):
        for mode in tqdm(['train', 'test']):
            print(f"loading the {mode} Data......")
            data_list = list()

            fin = open(self.file_name[self.dataset][mode], 'r', encoding='utf-8')
            lines = fin.readlines()
            fin.close()

            fin = open(self.file_name[self.dataset][mode] + '.graph', 'rb')
            idx2graph = pickle.load(fin)
            fin.close()

            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                context = lines[i].replace('$T$ ', '').strip()
                text = lines[i].replace('$T$', lines[i+1].strip()).strip()
                aspect = lines[i+1].strip()

                context_indices = self.tokenizer.text_to_sequence(context)
                text_indices = self.tokenizer.text_to_sequence(text)
                left_indices = self.tokenizer.text_to_sequence(text_left)
                aspect_indices = self.tokenizer.text_to_sequence(aspect)
                self.opt.max_len = max(self.opt.max_len, len(text_indices))
                polarity = int(lines[i+2].strip()) + 1
                dependency_graph = idx2graph[i]

                data = {
                    'text': text,
                    'context_indices': context_indices,
                    'text_indices': text_indices,
                    'left_indices': left_indices,
                    'aspect_indices': aspect_indices,
                    'polarity': polarity,
                    'dependency_graph': dependency_graph
                }

                data_list.append(data)

            if mode is 'train':
                # random.shuffle(data_list)
                # train_id = int(len(data_list) * 0.8)
                self.all_data['train'] = data_list
                # self.all_data['val'] = data_list[train_id:]
            else:
                self.all_data['test'] = data_list

    def build_dataloader(self):
        train_dataloader = BucketIterator(data=self.all_data['train'], batch_size=self.opt.batchsize,
                                          length=self.opt.max_len, shuffle=True)
        # val_dataloader = BucketIterator(data=self.all_data['val'], batch_size=self.opt.batchsize, shuffle=False)
        test_dataloader = BucketIterator(data=self.all_data['test'], batch_size=self.opt.batchsize,
                                         length=self.opt.max_len, shuffle=False)
        return train_dataloader, test_dataloader

    def _init(self):
        self.load_text(self.opt.dataset)
        self.load_dict()
        self.embedding_matrix = build_embedding_matrix(self.tokenizer.word2idx, self.embedding_dim, self.dataset)
        self.load_data()

'''
class GraphDataset(Dataset):
    def __init__(self, all_data, mode):
        # all_data 是一个字典['train', 'val', 'test'] 每一个key里面是一个list，每一个list里面又是一个字典
        self.all_data = all_data
        self.mode = mode
        assert mode in ['train', 'val', 'test'], "please choose the mode from train/val/test"
        self.data = self.pad_data(all_data[mode])
        self.cols = ['context_indices', 'text_indices', 'left_indices', 'aspect_indices', 'polarity', 'dependency_graph']

    def pad_data(self, data):
        text_list = []
        batch_text_indices = []
        batch_context_indices = []
        batch_aspect_indices = []
        batch_left_indices = []
        batch_polarity = []
        batch_dependency_graph = []
        # 这边是所有都统一按照文本的最大长度来填充，其实按照其他的做法，也许按照当前类型词的最大长度来填充即可？
        max_len = max([len(t['text_indices']) for t in data])
        for item in data:
            text_indices, context_indices, aspect_indices, left_indices, polarity, dependency_graph, text = \
                item['text_indices'], item['context_indices'], item['aspect_indices'], item['left_indices'],\
                item['polarity'], item['dependency_graph'], item['text']
            text_padding = [0] * (max_len - len(text_indices))
            context_padding = [0] * (max_len - len(context_indices))
            aspect_padding = [0] * (max_len - len(aspect_indices))
            left_padding = [0] * (max_len - len(left_indices))
            text_list.append(text)
            batch_text_indices.append(text_indices + text_padding)
            batch_context_indices.append(context_indices + context_padding)
            batch_aspect_indices.append(aspect_indices + aspect_padding)
            batch_left_indices.append(left_indices + left_padding)
            batch_polarity.append(polarity)
            batch_dependency_graph.append(np.pad(dependency_graph, \
                ((0, max_len-len(text_indices)), (0, max_len-len(text_indices))), 'constant'))
        return {
            'max_len': max_len,
            'text': text_list,
            'text_indices': torch.tensor(batch_text_indices),
            'context_indices': torch.tensor(batch_context_indices), \
            'aspect_indices': torch.tensor(batch_aspect_indices),
            'left_indices': torch.tensor(batch_left_indices),
            'polarity': torch.tensor(batch_polarity),
            'dependency_graph': torch.tensor(batch_dependency_graph),
            }
'''

class BucketIterator(object):
    def __init__(self, data, batch_size, length, sort_key='text_indices', shuffle=True, sort=True):
        self.length = length
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        # 计算总共需要多少个batch
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            # 根据文本的长度进行排序
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i*batch_size: (i+1)*batch_size], self.length))
        return batches

    def pad_data(self, batch_data, max_len):
        batch_text = []
        batch_text_indices = []
        batch_context_indices = []
        batch_aspect_indices = []
        batch_left_indices = []
        batch_polarity = []
        batch_dependency_graph = []
        # 这边是所有都统一按照文本的最大长度来填充，其实按照其他的做法，也许按照当前类型词的最大长度来填充即可？
        # max_len = max([len(t[self.sort_key]) for t in batch_data])
        for item in batch_data:
            text_indices, context_indices, aspect_indices, left_indices, polarity, dependency_graph, text = \
                item['text_indices'], item['context_indices'], item['aspect_indices'], item['left_indices'],\
                item['polarity'], item['dependency_graph'], item['text']
            text_padding = [0] * (max_len - len(text_indices))
            context_padding = [0] * (max_len - len(context_indices))
            aspect_padding = [0] * (max_len - len(aspect_indices))
            left_padding = [0] * (max_len - len(left_indices))
            batch_text.append(text)
            batch_text_indices.append(text_indices + text_padding)
            batch_context_indices.append(context_indices + context_padding)
            batch_aspect_indices.append(aspect_indices + aspect_padding)
            batch_left_indices.append(left_indices + left_padding)
            batch_polarity.append(polarity)
            batch_dependency_graph.append(np.pad(dependency_graph,
                                                 ((0, max_len-len(text_indices)), (0, max_len-len(text_indices))), 'constant'))
        return {
            'text': batch_text,
            'text_indices': torch.tensor(batch_text_indices),
            'context_indices': torch.tensor(batch_context_indices),
            'aspect_indices': torch.tensor(batch_aspect_indices),
            'left_indices': torch.tensor(batch_left_indices),
            'polarity': torch.tensor(batch_polarity),
            'dependency_graph': torch.tensor(batch_dependency_graph),
        }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]
