import fitlog
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from copy import deepcopy
from layers.dynamic_rnn import DynamicLSTM
from Policy import Policy_

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        # (bs , seq_len, hidden)    (bs, seq_len, seq_len)      bs, seq_len, hidden_2
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN(nn.Module):
    def __init__(self, opt):
        super(GCN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(self.opt.embedding_matrix, dtype=torch.float))
        self.lstm = DynamicLSTM(self.opt.embed_dim, self.opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gcn1 = GraphConvolution(2*self.opt.hidden_dim, 2*self.opt.hidden_dim)
        # self.gcn2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.fc = nn.Linear(2*self.opt.hidden_dim, self.opt.polarity)
        self.text_dropout = nn.Dropout(self.opt.dropout)

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i, 0]):
                weight[i].append(1-(aspect_double_idx[i, 0]-j)/context_len)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1]+1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i, 1]+1, text_len[i]):
                weight[i].append(1-(j-aspect_double_idx[i, 1])/context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight, dtype=torch.float).unsqueeze(2).to(self.opt.device)
        return weight*x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i,0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i,1]+1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(2).to(self.opt.device)
        return mask*x

    def forward(self, text_indices, aspect_indices, left_indices, adj):
        # adj放到cuda上
        adj = adj.to(self.opt.device)

        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)

        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len + aspect_len - 1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        text = self.text_dropout(text)
        text_out, (_, _) = self.lstm(text, text_len)  # .cpu()torch1.7的会出问题，要把text_len转到cpu上
        if text_out.shape[1] != text.shape[1]:
            zero_pad = torch.zeros((text_out.shape[0], (text.shape[1] - text_out.shape[1]), 2*self.opt.hidden_dim)).type_as(text_out)
            text_out = torch.cat([text_out, zero_pad], dim=1)
        out = F.relu(self.gcn1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        # out = F.relu(self.gcn2(self.position_weight(out, aspect_double_idx, text_len, aspect_len), adj))
        out = self.mask(out, aspect_double_idx)

        alpha_mat = torch.matmul(out, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        out = torch.matmul(alpha, text_out).squeeze(1)

        output = self.fc(out)
        return output




class Policy_GCN(nn.Module):
    def __init__(self, opt):
        super(Policy_GCN, self).__init__()
        self.opt = opt
        self.opt.cols = ['context_indices', 'text_indices', 'left_indices', 'aspect_indices', 'polarity', 'dependency_graph']
        self.activenet = GCN(self.opt).to(self.opt.device)
        self.targetnet = GCN(self.opt).to(self.opt.device)
        self._reset_params()
        self.assign_active_network()
        self.policy = Policy_(self.opt)

        self.opt_active = torch.optim.Adam(filter(lambda p: p.requires_grad, self.activenet.parameters()), self.opt.lr)
        self.opt_target = torch.optim.Adam(filter(lambda p: p.requires_grad, self.targetnet.parameters()), self.opt.lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    def warmup_train(self, dataloader):
        train_dataloader, test_dataloader = dataloader
        for epoch in range(self.opt.warmup_epoch):
            loss_epoch = 0.
            for (i_dx, datas) in enumerate(train_dataloader):
                self.targetnet.train()
                self.opt_active.zero_grad()
                self.opt_target.zero_grad()

                context_i, text_i, left_i, aspect_i, polirity, dependency_i = [datas[col].to(self.opt.device) for col in
                                                                               self.opt.cols]
                output = self.targetnet(text_i, aspect_i, left_i, dependency_i)
                loss = self.criterion(output, polirity)
                loss.backward()

                self.assign_active_network_gradients()
                self.opt_active.step()

                loss_epoch += loss.item()
                if i_dx % 10 == 0:
                    self.assign_target_network()

            self.assign_target_network()
            print(f"Warmup Epoch: {epoch}  Loss: {loss_epoch}")

        val_acc = 0.
        total_num = 0
        for (i_dx, datas) in enumerate(test_dataloader):
            self.targetnet.eval()
            context_i, text_i, left_i, aspect_i, polirity, dependency_i = [datas[col].to(self.opt.device) for col in
                                                                           self.opt.cols]
            output = self.targetnet(text_i, aspect_i, left_i, dependency_i)

            val_acc += (torch.argmax(output, -1) == polirity).sum().item()
            total_num += len(polirity)

        print(f"Warmup Done! the Acc on validation dataset: {val_acc / total_num}")
        self.assign_active_network()

    '''已弃用'''
    def train_gcn(self, policynet, dataloader):
        train_loader, test_loader = dataloader
        best_test_acc = 0.
        best_test_f1 = 0.

        for i in range(self.opt.epoch):
            '''
            train_acc = 0.
            total_num = 0
            total_loss = 0.
            for (i_dx, datas) in enumerate(train_loader):
                self.opt_target.zero_grad()
                policynet.train(False)
                self.targetnet.train()
                context_i, text_i, left_i, aspect_i, polirity, dependency_i = [datas[col].to(self.opt.device) for col in
                                                                               self.opt.cols]
                text = datas['text']
                assert len(text_i) == len(text)
                out = policynet(text_i)
                # out.shape([64, 9, k]) actionlist.shape: (64, 9)
                actions = self.policy.choose_action_batches(out, self.policy.episodes[self.policy.step], Random=True)
                matrixs = []
                for (idx, sen) in enumerate(text_i):
                    # print(text[idx])
                    tokens = self.opt.tokenizer(text[idx])
                    words = text[idx].split()
                    assert len(words) == len(list(tokens))
                    matrix = np.zeros((len(sen), len(sen))).astype('float32')
                    for token_i, token in enumerate(sen):
                        if token_i >= len(words):
                            break
                        matrix = Policy_.get_adj_from_action_words(tokens, token_i, actions[idx][token_i], matrix)
                    matrixs.append(matrix)
                matrixs = torch.FloatTensor(matrixs)
                output = self.targetnet(text_i, aspect_i, left_i, matrixs)

                loss = self.criterion(output, polirity)
                loss.backward()

                total_loss += loss.item()
                train_acc += (torch.argmax(output, -1) == polirity).sum().item()
                total_num += len(polirity)

            print(f"Epoch: {i+1}/{self.opt.epoch}: GCN training_acc: {train_acc / total_num}  GCN training_loss: {total_loss / total_num}")
            '''

            test_acc = 0.
            test_num = 0
            test_loss = 0.
            test_targets_all, test_outputs_all = None, None
            for (i_dx, datas) in enumerate(test_loader):
                self.targetnet.eval()
                context_i, text_i, left_i, aspect_i, polirity, dependency_i = [datas[col].to(self.opt.device) for col in
                                                                               self.opt.cols]
                text = datas['text']
                assert len(text_i) == len(text)

                # out = policynet(text_i)
                # out.shape([64, 9, k]) actionlist.shape: (64, 9)
                # actions = self.policy.choose_action_batches(out, self.policy.episodes[self.policy.step], Random=True)
                # matrixs = []
                # for (idx, sen) in enumerate(text_i):
                #     # print(text[idx])
                #     tokens = self.opt.tokenizer(text[idx])
                #     words = text[idx].split()
                #     assert len(words) == len(list(tokens))
                #     matrix = np.zeros((len(sen), len(sen))).astype('float32')
                #     for token_i, token in enumerate(sen):
                #         if token_i >= len(words):
                #             break
                #         matrix = Policy_.get_adj_from_action_words(tokens, token_i, actions[idx][token_i], matrix)
                #     matrixs.append(matrix)

                matrixs = []
                for (idx, sen) in enumerate(text_i):
                    tokens = self.opt.tokenizer(text[idx])
                    words = text[idx].split()
                    assert len(words) == len(list(tokens))
                    matrix = np.zeros((len(sen), len(sen))).astype('float32')
                    for token_i, token in enumerate(sen):
                        if token_i >= len(words):
                            break
                        output = policynet(token.view(1, 1), torch.FloatTensor(matrix).to(self.opt.device))
                        # 根据概率选择对应的动作
                        action = self.policy.choose_action(output, self.policy.episodes[self.policy.step])
                        matrix = Policy_.get_adj_from_action_words(tokens, token_i, action, matrix)
                    matrixs.append(matrix)


                matrixs = torch.FloatTensor(matrixs)
                output = self.targetnet(text_i, aspect_i, left_i, matrixs)

                loss = self.criterion(output, polirity)
                # loss.backward()  # test的时候不用backward

                test_loss += loss.item()
                test_acc += (torch.argmax(output, -1) == polirity).sum().item()
                test_num += len(polirity)

                if test_targets_all is None:
                    test_targets_all = polirity
                    test_outputs_all = output
                else:
                    test_targets_all = torch.cat((test_targets_all, polirity), dim=0)
                    test_outputs_all = torch.cat((test_outputs_all, output), dim=0)

            test_f1 = metrics.f1_score(test_targets_all.cpu(), torch.argmax(test_outputs_all, -1).cpu(), labels=[0, 1, 2],
                                      average='macro')
            print(f"Epoch: {i + 1}/{self.opt.epoch}: GCN test_acc: {test_acc / test_num} GCN test_f1: {test_f1} GCN test_loss: {test_loss / test_num}")

            if (test_acc / test_num) > best_test_acc:
                best_test_acc = test_acc / test_num
                best_test_f1 = test_f1
                fitlog.add_best_metric({"test": {"best_Acc": best_test_acc * 100}})
                fitlog.add_best_metric({"test": {"best_f1": best_test_f1 * 100}})
                print('best test acc!')

    def assign_active_network(self):
        params = []
        for name, x in self.targetnet.named_parameters():
            params.append(x)
        i = 0
        for name, x in self.activenet.named_parameters():
            x.data = deepcopy(params[i].data)
            i += 1

    def assign_target_network(self):
        # 将active网络中的参数取出并赋值给target网络
        # 此处model.named_parameters可以之后借鉴下
        params = []
        for name, x in self.activenet.named_parameters():
            params.append(x)
        i = 0
        for name, x in self.targetnet.named_parameters():
            x.data = deepcopy(params[i].data)
            i += 1

    def assign_active_network_gradients(self):
        params = []
        for name, x in self.targetnet.named_parameters():
            params.append(x)
        i = 0
        for name, x in self.activenet.named_parameters():
            x.grad = deepcopy(params[i].grad)
            i += 1
        for name, x in self.targetnet.named_parameters():
            x.grad = None

    def _reset_params(self):
        for p in self.targetnet.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    torch.nn.init.xavier_uniform_(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

        for p in self.activenet.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    torch.nn.init.xavier_uniform_(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)