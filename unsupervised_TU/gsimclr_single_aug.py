import os.path
import os.path as osp
import datetime
import random
import torch
from matplotlib import pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
# from core.encoders import *

# from torch_geometric.datasets import TUDataset
from aug import TUDataset_aug as TUDataset
from torch_geometric.loader import DataLoader
import sys
import json
from torch import optim

from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *
from gin import Encoder
from evaluate_embedding import evaluate_embedding
from model import *

from arguments import arg_parse
from torch_geometric.transforms import Constant
import pdb

from AGC import GGCR_calculate


class GcnInfomax(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
        super(GcnInfomax, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = args.prior

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

        self.local_d = FF(self.embedding_dim)
        self.global_d = FF(self.embedding_dim)
        # self.local_d = MI1x1ConvNet(self.embedding_dim, mi_units)
        # self.global_d = MIFCNet(self.embedding_dim, mi_units)

        if self.prior:
            self.prior_d = PriorDiscriminator(self.embedding_dim)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, num_graphs):

        # batch_size = data.num_graphs
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        y, M = self.encoder(x, edge_index, batch)

        g_enc = self.global_d(y)
        l_enc = self.local_d(M)

        mode = 'fd'
        measure = 'JSD'
        local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure)

        if self.prior:
            prior = torch.rand_like(y)
            term_a = torch.log(self.prior_d(prior)).mean()
            term_b = torch.log(1.0 - self.prior_d(y)).mean()
            PRIOR = - (term_a + term_b) * self.gamma
        else:
            PRIOR = 0

        return local_global_loss + PRIOR


class simclr(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
        super(simclr, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = args.prior

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, num_graphs):

        # batch_size = data.num_graphs
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        y, M = self.encoder(x, edge_index, batch)

        y = self.proj_head(y)

        return y

    def loss_cal(self, x, x_aug):

        T = 0.2  # init 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss


import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':

    args = arg_parse()
    setup_seed(args.seed)

    accuracies = {'val': [], 'test': []}
    epochs = 20
    log_interval = 10
    batch_size = args.batch_size
    lr = args.lr
    DS = args.DS
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

    # lists storing agc and acc
    degree_list = []
    agc_list = []
    acc_list = []
    # value interval of augmentation strengths
    space = 0.2 / 10
    print('{},{},{},{},{}\n'.format(args.DS, args.aug, args.batch_size, args.num_gc_layers, args.seed))
    time = str(datetime.datetime.now())
    for i in range(10):
        if args.aug == 'subgraph':
            degree = i * space + 0.1
        else:
            degree = i * space
        print('degree:', degree)
        if args.aug == 'dnodes':
            dataset = TUDataset(path, name=DS, aug=args.aug, dnodes_degree=degree).shuffle()
        elif args.aug == 'pedges':
            dataset = TUDataset(path, name=DS, aug=args.aug, pedges_degree=degree).shuffle()
        elif args.aug == 'subgraph':
            dataset = TUDataset(path, name=DS, aug=args.aug, subgraph_degree=degree).shuffle()
        else:
            dataset = TUDataset(path, name=DS, aug=args.aug, mask_nodes_degree=degree).shuffle()
        dataset_eval = TUDataset(path, name=DS, aug='none').shuffle()
        print(len(dataset))
        print(dataset.get_num_feature())
        try:
            dataset_num_features = dataset.get_num_feature()
        except:
            dataset_num_features = 1

        dataloader = DataLoader(dataset, batch_size=batch_size)
        dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = simclr(args.hidden_dim, args.num_gc_layers).to(device)
        # print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        print('================')
        print('lr: {}'.format(lr))
        print('num_features: {}'.format(dataset_num_features))
        print('hidden_dim: {}'.format(args.hidden_dim))
        print('num_gc_layers: {}'.format(args.num_gc_layers))
        print('================')

        model.eval()
        emb, y = model.encoder.get_embeddings(dataloader_eval)
        # print(emb.shape, y.shape)

        """
        acc_val, acc = evaluate_embedding(emb, y)
        accuracies['val'].append(acc_val)
        accuracies['test'].append(acc)
        """

        total_features = torch.tensor([]).to(device)
        ggcr_init = 0
        ggcr_final = 0
        agc = 0
        for epoch in range(1, epochs + 1):
            loss_all = 0
            model.train()
            # calculate ggcr_init
            if epoch == 2:
                ggcr_init = GGCR_calculate(total_features, 2, batch_size, device)
                total_features = torch.tensor([]).to(device)
                print('-----ggcr_init:', ggcr_init, '-----\n')
            for data in dataloader:

                # print('start')
                data, data_aug = data
                optimizer.zero_grad()

                node_num, _ = data.x.size()
                data = data.to(device)
                x = model(data.x, data.edge_index, data.batch, data.num_graphs)

                if args.aug == 'dnodes' or args.aug == 'subgraph' or args.aug == 'random2' or args.aug == 'random3' or args.aug == 'random4':
                    # node_num_aug, _ = data_aug.x.size()
                    edge_idx = data_aug.edge_index.numpy()
                    _, edge_num = edge_idx.shape
                    idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

                    node_num_aug = len(idx_not_missing)
                    data_aug.x = data_aug.x[idx_not_missing]

                    data_aug.batch = data.batch[idx_not_missing]
                    idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}
                    edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if
                                not edge_idx[0, n] == edge_idx[1, n]]
                    data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)

                data_aug = data_aug.to(device)

                '''
                print(data.edge_index)
                print(data.edge_index.size())
                print(data_aug.edge_index)
                print(data_aug.edge_index.size())
                print(data.x.size())
                print(data_aug.x.size())
                print(data.batch.size())
                print(data_aug.batch.size())
                pdb.set_trace()
                '''

                x_aug = model(data_aug.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs)

                # concate features of raw graphs and augmented graphs
                if epoch == 1 or epoch == epochs:
                    total_features = torch.cat((total_features, x, x_aug), 0)

                # print(x)
                # print(x_aug)
                loss = model.loss_cal(x, x_aug)
                print(loss)
                loss_all += loss.item() * data.num_graphs
                loss.backward()
                optimizer.step()
                # print('batch')
            print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))

            if epoch % log_interval == 0:
                model.eval()
                emb, y = model.encoder.get_embeddings(dataloader_eval)
                acc_val, acc = evaluate_embedding(emb, y)
                accuracies['val'].append(acc_val)
                accuracies['test'].append(acc)
                # print(accuracies['val'][-1], accuracies['test'][-1])

        # calculate ggcr_final and agc
        ggcr_final = GGCR_calculate(total_features, 2, batch_size, device)
        agc = (1 - float(ggcr_final)) / (1 - float(ggcr_init))
        print('-----ggcr_final:', ggcr_final, '-----\n')
        print('-----agc:', agc, '-----\n')

        degree_list.append(degree)
        agc_list.append(agc)
        acc_list.append(np.mean(accuracies['test']))

        tpe = ('local' if args.local else '') + ('prior' if args.prior else '')
        file_path = 'logs/' + args.DS + '_' + args.aug + '_' + str(
            args.batch_size) + '_' + str(args.num_gc_layers) + '_' + str(args.seed) + '_' + time + '.txt'

        if not os.path.exists(file_path):
            open(file_path, 'a').close()
        with open(file_path, 'a+') as f:
            s = json.dumps(accuracies)
            f.write('{}\n'.format(datetime.datetime.now()))
            f.write('{},{},{},{},{},{},{}\n'.format(args.DS, tpe, args.num_gc_layers, epochs, log_interval, lr, s))
            f.write('augment degree:{}\n'.format(degree))
            f.write('agc:{}\n'.format(agc))
            f.write('\n')
        accuracies = {'val': [], 'test': []}

    # draw the line chart of agc and acc
    dic = {}
    for i in range(len(degree_list)):
        dic[degree_list[i]] = agc_list[i]
    x = sorted(dic)
    agc = []
    for item in x:
        agc.append(dic[item])

    dic = {}
    for i in range(len(degree_list)):
        dic[degree_list[i]] = acc_list[i]
    x = sorted(dic)
    acc = []
    for item in x:
        acc.append(dic[item])

    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x, agc, 'g-')
    ax2.plot(x, acc, 'b--')

    ax1.set_xlabel('aug strength')
    ax1.set_ylabel('agc', color='g')
    ax2.set_ylabel('acc', color='b')
    fig_path = 'figs/' + args.DS + '_' + args.aug + '_' + str(
        args.batch_size) + '_' + str(args.num_gc_layers) + '_' + str(args.seed) + '_' + str(
        datetime.datetime.now()) + '.png'
    plt.savefig(fig_path)
    plt.show()
