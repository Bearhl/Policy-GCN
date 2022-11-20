import math
import copy
import random
import queue
import torch
import fitlog
import numpy as np
import torch.nn as nn
from sklearn import metrics
from collections import Counter

from tqdm import tqdm

class PolicyNet(nn.Module):
    """
    假设输入的状态是一个单词的隐藏层表示，输出是当前句子应该执行多少跳
    """
    def __init__(self, opt):
        super(PolicyNet, self).__init__()
        self.opt = opt

        self.emb = nn.Embedding.from_pretrained(torch.tensor(self.opt.embedding_matrix, dtype=torch.float))
        self.W1 = nn.Parameter(torch.FloatTensor(2*self.opt.embed_dim, self.opt.hidden_dim))
        self.b1 = nn.Parameter(torch.FloatTensor(self.opt.hidden_dim, ))

        self.W2 = nn.Parameter(torch.FloatTensor(1, self.opt.max_len))
        self.W3 = nn.Parameter(torch.FloatTensor(self.opt.max_len, self.opt.hidden_dim))
        self.b2 = nn.Parameter(torch.FloatTensor(self.opt.hidden_dim, ))

        self.W4 = nn.Parameter(torch.FloatTensor(2*self.opt.hidden_dim, self.opt.max_jump))

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=-1)

    def forward(self, x, adj, aspect):
        # (1, 1)  (79, 79) (1, 79)
        x = self.emb(x)
        # emb_matrix第一行为0，亦padding之后经过emb也是0
        aspect_len = torch.sum(aspect != 0, dim=-1)
        aspect = self.emb(aspect)
        aspect = torch.sum(aspect, dim=1) / aspect_len.float()
        aspect_x = torch.cat([aspect.unsqueeze(1), x], dim=-1)
        x_out = self.relu(torch.matmul(aspect_x, self.W1) + self.b1).squeeze(1)

        adj_out = self.relu(torch.matmul(self.W2, adj))
        adj_out = self.relu(torch.matmul(adj_out, self.W3) + self.b2)

        out = torch.cat([x_out, adj_out], dim=-1)
        out = torch.matmul(out, self.W4)
        result = self.soft(out)
        return result


class Policy_():
    def __init__(self, opt):
        # super(Policy_, self).__init__()
        self.opt = opt
        self.seed = self.opt.seed
        self.step = 0
        self.actor_target = PolicyNet(self.opt).to(self.opt.device)
        self.actor_eval = PolicyNet(self.opt).to(self.opt.device)
        self._reset_params()
        self.update_target_network()
        self.episodes = np.linspace(self.opt.epsilon_start, self.opt.epsilon_end, self.opt.epsilon_decay_steps)

        self.actor_target_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.actor_target.parameters()))
        self.actor_eval_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.actor_eval.parameters()))

    def train(self, dataloader, gcn):
        gcn_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gcn.parameters()))
        best_val_acc, best_val_f1 = 0., 0.
        loss_fn = nn.CrossEntropyLoss()
        print(gcn)
        train_dataloader, test_dataloader = dataloader

        for episode in range(self.opt.max_episodes):
            for (i_dx, datas) in enumerate(tqdm(train_dataloader)):

                context_i, text_i, left_i, aspect_i, polirity, dependency_i = [datas[col].to(self.opt.device) for col
                                                                               in self.opt.cols]
                text = datas['text']
                assert len(text_i) == len(text)
                # 遍历每一个句子
                for (idx, sen) in enumerate(text_i):
                    tokens = self.opt.tokenizer(text[idx])
                    words = text[idx].split()
                    assert len(words) == len(list(tokens))

                    aveloss = 0.
                    losslist = list()
                    statelist = list()
                    actionlist = list()
                    outputlist = list()

                    base_matrix = np.zeros((len(sen), len(sen))).astype('float32')
                    for num in range(len(words)):
                        base_matrix[num][num] = 1

                    for samplecnt in range(self.opt.samplecnt):
                        # 存储每一个单词的的动作
                        actiontemp = list()
                        matrixtemp = list()
                        outputtemp = list()

                        matrix = base_matrix
                        # 遍历每一个单词
                        for token_i, token in enumerate(sen):
                            if token_i >= len(words):
                                break
                            output = self.actor_target(token.view(1, 1), torch.FloatTensor(matrix).to(self.opt.device), aspect_i[idx].view(1, -1))
                            # 根据概率选择对应的动作
                            actiontemp.append(self.choose_action(output, self.episodes[self.step]))
                            matrix = self.get_adj_from_action_words(tokens, token_i, actiontemp[-1], matrix)
                            matrixtemp.append(matrix)
                            outputtemp.append(output)
                        # 保存结果
                        statelist.append([sen, matrixtemp, aspect_i[idx].view(1, -1)])
                        actionlist.append(actiontemp)

                        outputlist.append(outputtemp)

                        length = len(actiontemp)
                        Zerolength = Counter(actiontemp)

                        gcn.train()
                        out = gcn(text_i[idx].unsqueeze(0), aspect_i[idx].unsqueeze(0), left_i[idx].unsqueeze(0),
                                  torch.FloatTensor(statelist[-1][1][-1]).unsqueeze(0))
                        loss_ = loss_fn(out, polirity[idx].unsqueeze(0))
                        # 增加损失 避免模型生成较多的0action
                        if Zerolength.get(0) != None:
                            loss_ += (float(Zerolength.get(0)) / length) ** 2
                        aveloss += loss_.item()
                        losslist.append(loss_)

                    if self.step < self.opt.epsilon_decay_steps - 1:
                        self.step += 1

                    aveloss /= self.opt.samplecnt

                    grad0, grad1, grad2, grad3, grad4, grad5 = 0, 0, 0, 0, 0, 0
                    for i in range(self.opt.samplecnt):
                        # print(losslist[i].item() - aveloss + 1e-9)
                        for pos in range(len(actionlist[i])):
                            rr = [0] * self.opt.max_jump
                            rr[actionlist[i][pos]] = (losslist[i].item() - aveloss + 1e-9) * self.opt.alpha
                            # g = self.actor_get_gradient(statelist[i][0][pos], torch.FloatTensor(statelist[i][1][pos]), rr, statelist[i][2])
                            g = self.actor_get_gradient_modify(outputlist[i][pos], rr)

                            # for t in range(len(g)):
                            #     exec('grad{} += {}'.format(t, g[t]))
                            grad0 += g[0]
                            grad1 += g[1]
                            grad2 += g[2]
                            grad3 += g[3]
                            grad4 += g[4]
                            grad5 += g[5]

                    self.actor_target_optimizer.zero_grad()
                    self.actor_eval_optimizer.zero_grad()

                    self.assign_eval_network_gradients(grad0, grad1, grad2, grad3, grad4, grad5)
                    self.actor_eval_optimizer.step()

                    # GCN网络的训练
                    gcn_optimizer.zero_grad()

                    gcn_loss = sum(losslist) / len(losslist)

                    gcn_loss.backward()
                    gcn_optimizer.step()

                # 遍历完一个batchsize的数据后更新target网络
                self.update_target_network()

            # 每遍历一次train loader 就在GCN上验证一下正确率
            print("Episode eval...")
            if self.opt.debug:
                val_acc, val_f1 = self.eval_policy_debug(gcn, test_dataloader, episode)
            else:
                val_acc, val_f1 = self.eval_policy(gcn, test_dataloader)
            print(f"Episodes: {episode+1} / {self.opt.max_episodes} the val acc is {val_acc}, the val f1 is {val_f1}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                fitlog.add_best_metric({"eval": {"best_Acc": best_val_acc * 100}})
                fitlog.add_best_metric({"eval": {"best_f1": best_val_f1 * 100}})
                print('best val acc!')

    def choose_action_batches(self, pred, epsilon, Random=True):
        # pred shape: (bs, seq_len, k) output shape: (bs, seq_len)
        # 如果有nan 先将其替换为1 这个bug待解决
        pred = torch.where(torch.isnan(pred), torch.full_like(pred, 1), pred)
        final_action = []
        for i in range(pred.shape[0]):
            if Random:
                if random.random() > epsilon:
                    actions = list(torch.multinomial(pred[i], 1, replacement=False).view(-1).cpu().numpy() + 1)
                else:
                    actions = [random.randint(0, self.opt.max_jump-1) + 1 for _ in range(pred[i].shape[0])]
            else:
                actions = np.array(torch.argmax(pred[i], dim=-1)) + 1

            final_action.append(actions)

        return final_action

    def choose_action(self, pred, epsilon, Random=True):
        # seed问题待解决
        pred = torch.where(torch.isnan(pred), torch.full_like(pred, 1), pred)
        if Random:
            self.seed += 1
            random.seed(self.seed)
            if random.random() > epsilon:
                action = int(torch.multinomial(pred, 1, replacement=False).view(-1))
            else:
                action = random.randint(0, self.opt.max_jump-1)
        else:
            action = int((torch.argmax(pred.cpu(), dim=-1).numpy()))
        return action

    @staticmethod
    def get_adj_from_action_words(tokens, idx, action, matrix):
        token = tokens[idx]
        # 用来存储child迭代器
        origin_queue = queue.Queue()
        # 用来存储中间结果
        temp_queue = queue.Queue()
        output = list()
        origin_queue.put(token)
        for i in range(action):
            while not origin_queue.empty():
                for child in origin_queue.get().children:
                    output.append(child)
                    temp_queue.put(child)
            while not temp_queue.empty():
                origin_queue.put(temp_queue.get())

        # 对matrix进行标注
        for t in output:
            matrix[token.i][t.i] = 1
            matrix[t.i][token.i] = 1
        return matrix

    def actor_get_gradient(self, x, adj, reward, aspect):
        out = self.actor_target(x.view(1, 1), adj.to(self.opt.device), aspect)
        # out.shape:(1, max_jump)
        # 因为此处需要从中抽出非0的索引
        logout = torch.log(out).view(-1)
        index = np.nonzero(np.array(reward))
        index = int(index[0])

        grad = torch.autograd.grad(logout[index].view(-1), filter(lambda p: p.requires_grad, self.actor_target.parameters()))

        for i in range(len(grad)):
            grad[i].data = grad[i].data * reward[index]

        return grad


    def actor_get_gradient_modify(self, out, reward):
        logout = torch.log(out).view(-1)
        index = np.nonzero(np.array(reward))
        index = int(index[0])

        grad = torch.autograd.grad(logout[index].view(-1), filter(lambda p: p.requires_grad, self.actor_target.parameters()))

        for i in range(len(grad)):
            grad[i].data = grad[i].data * reward[index]

        return grad

    def assign_eval_network_gradients(self, grad0, grad1, grad2, grad3, grad4, grad5):
        params = [grad0, grad1, grad2, grad3, grad4, grad5]
        i = 0
        for name, x in self.actor_eval.named_parameters():
            if name == 'emb.weight':
                continue
            x.grad = copy.deepcopy(params[i])
            i += 1

    def update_target_network(self):
        params = []
        for name, x in self.actor_eval.named_parameters():
            params.append(x)
        i = 0
        for name, x in self.actor_target.named_parameters():
            x.data = copy.deepcopy(params[i].data)
            i += 1

    def eval_policy(self, gcn, dataloader):
        val_acc = 0.
        val_targets_all, val_outputs_all = None, None
        total_num = 0
        for (i_dx, datas) in enumerate(dataloader):
            gcn.eval()
            context_i, text_i, left_i, aspect_i, polirity, dependency_i = [datas[col].to(self.opt.device) for col in
                                                                           self.opt.cols]

            text = datas['text']
            assert len(text_i) == len(text)

            matrixs = []
            for (idx, sen) in enumerate(text_i):
                tokens = self.opt.tokenizer(text[idx])
                words = text[idx].split()
                assert len(words) == len(list(tokens))
                matrix = np.zeros((len(sen), len(sen))).astype('float32')
                for token_i, token in enumerate(sen):
                    if token_i >= len(words):
                        break
                    output = self.actor_target(token.view(1, 1), torch.FloatTensor(matrix).to(self.opt.device), aspect_i[idx].view(1, -1))
                    # 根据概率选择对应的动作
                    action = self.choose_action(output, self.episodes[self.step], Random=False)
                    matrix = self.get_adj_from_action_words(tokens, token_i, action, matrix)
                matrixs.append(matrix)

            matrixs = torch.FloatTensor(matrixs)
            output = gcn(text_i, aspect_i, left_i, matrixs)
            val_acc += (torch.argmax(output, -1) == polirity).sum().item()
            total_num += len(polirity)

            if val_targets_all is None:
                val_targets_all = polirity
                val_outputs_all = output
            else:
                val_targets_all = torch.cat((val_targets_all, polirity), dim=0)
                val_outputs_all = torch.cat((val_outputs_all, output), dim=0)

        val_f1 = metrics.f1_score(val_targets_all.cpu(), torch.argmax(val_outputs_all, -1).cpu(), labels=[0, 1, 2],
                                   average='macro')
        return val_acc / total_num, val_f1

    def eval_policy_debug(self, gcn, dataloader, episode):
        val_acc = 0.
        val_targets_all, val_outputs_all = None, None
        total_num = 0
        import csv
        f = open(f'bad_case_logs/bad_case{episode}.csv', 'a+')
        fnames = ['word', 'actions', 'label', 'pred']
        writer = csv.DictWriter(f, fnames)
        writer.writeheader()
        for (i_dx, datas) in enumerate(dataloader):
            gcn.eval()
            context_i, text_i, left_i, aspect_i, polirity, dependency_i = [datas[col].to(self.opt.device) for col in
                                                                           self.opt.cols]

            text = datas['text']
            assert len(text_i) == len(text)

            matrixs = []
            actions = []
            for (idx, sen) in enumerate(text_i):
                action_temp = []
                tokens = self.opt.tokenizer(text[idx])
                words = text[idx].split()
                assert len(words) == len(list(tokens))
                matrix = np.zeros((len(sen), len(sen))).astype('float32')
                for num in range(len(words)):
                    matrix[num][num] = 1

                for token_i, token in enumerate(sen):
                    if token_i >= len(words):
                        break
                    # print(token, matrix, aspect_i, text[idx])
                    output = self.actor_target(token.view(1, 1), torch.FloatTensor(matrix).to(self.opt.device), aspect_i[idx].view(1, -1))
                    action = self.choose_action(output, self.episodes[self.step], Random=False)
                    action_temp.append(action)
                    matrix = self.get_adj_from_action_words(tokens, token_i, action, matrix)
                print(action_temp)
                matrixs.append(matrix)
                actions.append(action_temp)

            # for (idx, sen) in enumerate(text_i):
            #     # print(text[idx])
            #     tokens = self.opt.tokenizer(text[idx])
            #     words = text[idx].split()
            #     assert len(words) == len(list(tokens))
            #     matrix = np.zeros((len(sen), len(sen))).astype('float32')
            #     for token_i, token in enumerate(sen):
            #         if token_i >= len(words):
            #             break
            #         matrix = self.get_adj_from_action_words(tokens, token_i, actions[idx][token_i], matrix)
            #     matrixs.append(matrix)
            matrixs = torch.FloatTensor(matrixs)
            output = gcn(text_i, aspect_i, left_i, matrixs)

            for idx, i in enumerate(torch.argmax(output, -1).cpu().tolist()):
                if i != int(polirity[idx].cpu()):
                    writer.writerow({'word': text[idx], 'actions': actions[idx], 'label': polirity[idx], 'pred': output[idx]})

            val_acc += (torch.argmax(output, -1) == polirity).sum().item()
            total_num += len(polirity)

            if val_targets_all is None:
                val_targets_all = polirity
                val_outputs_all = output
            else:
                val_targets_all = torch.cat((val_targets_all, polirity), dim=0)
                val_outputs_all = torch.cat((val_outputs_all, output), dim=0)
        f.close()
        val_f1 = metrics.f1_score(val_targets_all.cpu(), torch.argmax(val_outputs_all, -1).cpu(), labels=[0, 1, 2],
                                   average='macro')
        return val_acc / total_num, val_f1

    def run(self, dataloader, gcn):
        self.train(dataloader, gcn)

    def _reset_params(self):
        for p in self.actor_target.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    torch.nn.init.xavier_uniform_(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

        for p in self.actor_eval.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    torch.nn.init.xavier_uniform_(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)
