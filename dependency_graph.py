import os
import spacy
import pickle
import numpy as np

from spacy.tokens import Doc

class WhitespaceTokenizer(object):
    # 重写spcy的分词（空格完成分词）
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


class dependency_process():
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.tokenizer = WhitespaceTokenizer(self.nlp.vocab)

    def dependency_adj(self, text):
        """
        基于一阶邻居的邻接矩阵
        """
        tokens = self.nlp(text)
        words = text.split()
        matrix = np.zeros((len(words), len(words))).astype('float32')
        assert len(words) == len(list(tokens))
        for token in tokens:
            # 节点本身置为1
            matrix[token.i][token.i] = 1
            for child in token.children:
                # 无向图
                matrix[token.i][child.i] = 1
                matrix[child.i][token.i] = 1

        return matrix

    def precess_dataset(self, dataset):
        idx2graph = dict()
        output = dataset + '.graph'
        with open(dataset, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                adj_matrix = self.dependency_adj(text_left + ' ' + aspect + ' ' + text_right)
                idx2graph[i] = adj_matrix
            with open(output, 'wb') as fout:
                pickle.dump(idx2graph, fout)


if __name__ == '__main__':
    DP = dependency_process()
    DP.precess_dataset(os.path.join('datasets', 'acl-14-short-data', 'train.raw'))
    DP.precess_dataset(os.path.join('datasets', 'acl-14-short-data', 'test.raw'))
    DP.precess_dataset(os.path.join('datasets', 'semeval14', 'restaurant_train.raw'))
    DP.precess_dataset(os.path.join('datasets', 'semeval14', 'restaurant_test.raw'))
    DP.precess_dataset(os.path.join('datasets', 'semeval14', 'laptop_train.raw'))
    DP.precess_dataset(os.path.join('datasets', 'semeval14', 'laptop_test.raw'))
    DP.precess_dataset(os.path.join('datasets', 'semeval15', 'restaurant_train.raw'))
    DP.precess_dataset(os.path.join('datasets', 'semeval15', 'restaurant_test.raw'))
    DP.precess_dataset(os.path.join('datasets', 'semeval16', 'restaurant_train.raw'))
    DP.precess_dataset(os.path.join('datasets', 'semeval16', 'restaurant_test.raw'))