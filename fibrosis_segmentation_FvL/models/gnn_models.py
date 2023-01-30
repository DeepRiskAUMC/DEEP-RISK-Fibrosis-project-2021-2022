import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool, InstanceNorm
from torch_geometric.data import Data, Dataset, DataLoader
import numpy as np

"""
The main structure of this code has been provided by dr. Erik Bekkers and Kylian Geijtenbeek.
While the code has been modified, I do not claim to be the full original author.
Based on the following paper:
E(n)-Equivariant Graph Neural Networks; https://arxiv.org/abs/2102.09844v1
"""


class Resblock(nn.Module):
    """
    Residual block with two conv-batchnorm-relu(-maxpool)-dropout layers.
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels/number of convolution filters
        pool (bool): If True, use max-pooling
        dropout (float): Dropout rate
        kernel_size (int): Convolution kernel size
    """
    def __init__(self,
        in_channels,
        out_channels,
        pool=False,
        dropout=0.1,
        kernel_size=5):
        super(Resblock, self).__init__()

        self.conv1 = nn.Sequential( nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                                    nn.BatchNorm1d(out_channels),
                                    nn.ReLU(),
                                    #nn.MaxPool1d(2, 2) if pool else nn.Identity(),
                                    nn.Dropout(dropout))
        self.conv2 = nn.Sequential( nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2),
                                    nn.BatchNorm1d(out_channels),
                                    nn.ReLU(),
                                    # nn.MaxPool1d(2, 2) if pool else nn.Identity(),
                                    nn.Dropout(dropout))

        self.skip_conv = nn.Sequential( nn.Conv1d(in_channels, out_channels, 1, padding=0)
                                        # nn.MaxPool1d(2, 2) if pool else nn.Identity()
                                        )

    def forward(self, x):
        """
        Forward propagation
        """
        out = self.conv1(x)
        out = self.conv2(out)
        skip = self.skip_conv(x)
        out = out + skip
        return out


class EGNN_Conv(MessagePassing):
    """
    Graph convolution layer for the E(N) equivariant message passing convolutional neural network.
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels/number of convolution filters
        dim (int): The "N" in the E(N) equivariance
        update_pos (bool): If True, update node positions
        infer_edges (bool): If True, infer graph edges
        recurrent (bool): If True, updates nodes by adding an update feature to
            the original representation. Otherwise, updates representation with
            update network output directly
        dist_info (bool): If True, uses distance information. If False, this is
            a normal GNN and not EGNN
        dropout (float): Dropout rate
        kernel_size (int): Convolution kernel size
    """

    def __init__(self, in_channels, out_channels, dim=2, update_pos=False, infer_edges=False, recurrent=True, dist_info=False, dropout=0.1, kernel_size=5, distance_measure='euclidean'):
        super(EGNN_Conv, self).__init__(node_dim=-2, aggr="add")
        self.distance_measure = distance_measure
        self.update_pos = update_pos
        self.infer_edges = infer_edges
        self.dim = dim
        self.recurrent = recurrent
        self.dist_info = dist_info

        self.message_net = nn.Sequential( Resblock(2*in_channels + int(dist_info)*3, out_channels, pool=False, dropout=dropout, kernel_size=kernel_size),
                                          Resblock(out_channels, out_channels, pool=False, dropout=dropout, kernel_size=kernel_size))

        self.update_net = nn.Sequential( Resblock(2*in_channels, out_channels, pool=False, dropout=dropout, kernel_size=kernel_size),
                                         Resblock(out_channels, out_channels, pool=True, dropout=dropout, kernel_size=kernel_size))

    def forward(self, x, pos, edge_index, batch):
        """ Propagate messages along edges """
        b, c, l = x.shape
        self.input_shape = (b, c, l)
        x, pos = self.propagate(edge_index, x=x.reshape(b, c*l), pos=pos)
        # x = self.norm(x, batch)
        return x, pos

    def message(self, x_i, x_j, pos_i, pos_j):
        """ Message according to eqs 3-4 in the paper """
        if self.distance_measure == 'euclidean': 
            d = (pos_i - pos_j).pow(2) # Use this formula for d for euclidean distance
        elif self.distance_measure == 'displacement':
            diff = pos_i - pos_j
            d = diff # use this formula for d for displacement vector
        else:
            raise ValueError(f'distance_measure {self.distance_measure} not recognized')
        
        num_edges = x_i.shape[0]
        b, c, l = self.input_shape
        x_i, x_j = x_i.reshape(num_edges, c, l), x_j.reshape(num_edges, c, l)

        x_ij = torch.cat((x_i, x_j), dim=1)
        if self.dist_info:
            B, C, L = x_ij.shape
            d_feature = torch.ones(B, 3, L).to(x_ij.device) * d.reshape(B, 3, 1)
            x_ij = torch.cat((x_ij, d_feature), dim=1)

        message = self.message_net(x_ij)

        b, c, l = message.shape
        self.input_shape = (b, c, l)
        message = message.reshape(b, c*l)

        if self.infer_edges:
            message = self.inf_net(message)*message

        if self.update_pos:
            pos_message = (pos_i - pos_j) * self.pos_net(message)
            # torch geometric does not support tuple outputs.
            message = torch.cat((pos_message, message), dim=-1)
        return message

    def update(self, message, x, pos):
        """ Update according to eq 6 in the paper """
        if self.update_pos:
            pos_message = message[:, :self.dim]
            message = message[:, self.dim:]

            pos += pos_message
        if self.recurrent:
            x += self.update_net(torch.cat((x, message), dim=-1))
        else:
            _, c, l = self.input_shape
            b = x.shape[0]
            message = message.reshape(b, c, l)
            x = x.reshape(b, c, l)
            y = torch.cat((x, message), dim=1)
            x = self.update_net(torch.cat((x, message), dim=1))
        return x, pos

class EGNN_Conv_Model(nn.Module):
    """
    E(N) equivariant message passing convolutional neural network.
    Args:
        in_channels (int): Number of input channels
        num_classes (int): Number of output classes
        hidden_channels (int): Number of channels/filters for the hidden layers
        N (int): Number of graph convolution layers
        update_pos (bool): If True, update node positions
        infer_edges (bool): If True, infer graph edges
        recurrent (bool): If True, updates nodes by adding an update feature to
            the original representation. Otherwise, updates representation with
            update network output directly
        dropout (float): Dropout rate
        dist_info (bool): If True, uses distance information. If False, this is
            a normal GNN and not EGNN
        hidden_units_fc (int): Number of hidden units in the dense classification hidden layer
        kernel_size (int): Convolution kernel size
    """
    def __init__(
        self,
        in_channels=1,
        num_classes=1,
        hidden_channels=8,
        N=4,
        update_pos=False,
        infer_edges=False,
        recurrent=False,
        dropout=0.1,
        dist_info=False,
        hidden_units_fc=16,
        kernel_size=5,
        distance_measure='euclidean'
    ):
        super(EGNN_Conv_Model, self).__init__()

        self.embedding = self.embedder = nn.Sequential(nn.Linear(in_channels, hidden_channels),
                                      nn.ReLU(),
                                      nn.Linear(hidden_channels, hidden_channels))

        self.layers = []

        for i in range(N):
            self.layers.append(EGNN_Conv(hidden_channels, hidden_channels, update_pos=update_pos,
                                    infer_edges=infer_edges, recurrent=recurrent, dist_info=dist_info, 
                                    dropout=dropout, kernel_size=kernel_size, distance_measure=distance_measure))

        self.layers = nn.ModuleList(self.layers)

        self.head_pre_pool = nn.Sequential( nn.Linear(hidden_channels, hidden_units_fc),
                                            nn.BatchNorm1d(hidden_units_fc),
                                            nn.ReLU(),
                                            nn.Dropout(dropout),
                                            nn.Linear(hidden_units_fc, num_classes))

    def forward(self, graph):
        """
        Forward propagation.
        """
        x, pos = graph.x, graph.pos

        # make sure tensor is 3 dimensional (B * channels * nodes)
        if x.dim() < 2:
            x = x.unsqueeze(-1)
        if x.dim() < 3:
            x = x.unsqueeze(-1)

        x = self.embedding(x)
        # x = self.embedding(x.unsqueeze(1))

        for layer in self.layers:
            x, pos = layer(x, pos, graph.edge_index, graph.batch)

        # Output head
        B, C, L = x.shape
        x = x.reshape(B, C*L)
        out = self.head_pre_pool(x)
        out = global_mean_pool(out, graph.batch).squeeze()
        return out

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

class MPNNLayer(MessagePassing):
    """ Message Passing Layer """
    def __init__(self, node_features, edge_features, hidden_features, out_features, aggr, act, b_norm, i_norm):
        super().__init__(aggr=aggr)

        message_net_list = [nn.Linear(2 * node_features + edge_features, hidden_features)]
        if b_norm:
            message_net_list.append(nn.BatchNorm1d(hidden_features))
        elif i_norm:
            message_net_list.append(nn.InstanceNorm1d(hidden_features))
        message_net_list.append(act())
        message_net_list.append(nn.Linear(hidden_features, hidden_features))
        self.message_net = nn.Sequential(*message_net_list)

        update_net_list = [nn.Linear(node_features + hidden_features, hidden_features)]
        if b_norm:
            update_net_list.append(nn.BatchNorm1d(hidden_features))
        elif i_norm:
            update_net_list.append(nn.InstanceNorm1d(hidden_features))
        update_net_list.append(act())
        update_net_list.append(nn.Linear(hidden_features, out_features))
        self.update_net = nn.Sequential(*update_net_list)
    

    def forward(self, x, edge_index, edge_attr=None):
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return x

    def message(self, x_i, x_j, edge_attr):
        """ Construct messages between nodes """
        input = [x_i, x_j] if edge_attr is None else [x_i, x_j, edge_attr]
        input = torch.cat(input, dim=-1)
        message = self.message_net(input)
        return message

    def update(self, message, x):
        """ Update node features """
        input = torch.cat((x, message), dim=-1)
        update = self.update_net(input)
        return update


class MPNN(nn.Module):
    """ Message Passing Neural Network """
    def __init__(self, node_features, edge_features, hidden_features, out_features, num_layers, aggr, act, pool='mean', b_norm=False, i_norm=False):
        super().__init__()

        self.embedder = nn.Sequential(nn.Linear(node_features, hidden_features),
                                      act(),
                                      nn.Linear(hidden_features, hidden_features))
    
        # layer = MPNNLayer(node_features=hidden_features, 
        #                     hidden_features=hidden_features, 
        #                     edge_features=0,  
        #                     out_features=hidden_features, 
        #                     aggr=aggr, 
        #                     act=act,
        #                     b_norm=b_norm,
        #                     i_norm=i_norm)

        layers = []
        for i in range(num_layers):
            layer = MPNNLayer(node_features=hidden_features, 
                            hidden_features=hidden_features, 
                            edge_features=edge_features,  
                            out_features=hidden_features, 
                            aggr=aggr, 
                            act=act,
                            b_norm=b_norm,
                            i_norm=i_norm)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        if pool is None:
            self.pooler = None
        elif pool == "add":
            self.pooler = global_add_pool
        elif pool == "mean":
            self.pooler = global_mean_pool
        else:
            raise Exception("Pooler not recognized")

        self.head = nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                  act(),
                                  nn.Linear(hidden_features, out_features))

    def forward(self, graph):
        x, edge_index, edge_attr, batch = graph.x, graph.edge_index, graph.edge_attr, graph.batch
        if x.dim() < 2:
            x = x.reshape(-1,1)
        x = self.embedder(x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr) 
        if self.pooler:
            x = self.pooler(x, batch)
        x = self.head(x)
        return x


    def get_pre_pool_rep(self, x, edge_index, edge_attr=None):
        with torch.no_grad():            
            x = self.embedder(x)
            for layer in self.layers:
                x = layer(x, edge_index, edge_attr)
        return x