from torch import nn, Tensor
from typing import Optional
from torch_geometric.typing import Adj, Size, OptTensor

import torch
from torch import Tensor
from torch.nn import Embedding
import torch.nn.functional as F

from torch_geometric.nn import Sequential
import torch_geometric.transforms as T
from torch.nn import Linear, MSELoss
from torch.optim import Adam
from torch_geometric.nn import BatchNorm

from torch_geometric.utils import softmax
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear, HeteroLinear

from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing

class EGATConvs(MessagePassing):
    """
    Edge graph attention networks 
     
    Parameters:
        in_channels: node input dim.
        out_channels: output dim.
        edge_dim: edge input dim.
        edge_attr_emb_dim: the embedding size of edge attr.
        heads: number of mul. attention
        concat: (True) concatenate the multi-head attentions
        root_weight: (True) add transformed root node features to 
                     the output
    """
    def __init__(self, in_channels: int, out_channels: int,
                 edge_dim: int, edge_attr_emb_dim: int,
                 heads: int = 3, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 root_weight: bool = True, bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'max')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.root_weight = root_weight

        self.lin_x = Linear(in_channels, out_channels)
        self.edge_attr_emb = Linear(edge_dim, edge_attr_emb_dim, bias=False)

        self.att = Linear(
            2 * out_channels + edge_attr_emb_dim,
            self.heads, bias=False)

        self.lin = Linear(out_channels + edge_attr_emb_dim, out_channels,
                          bias=bias)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.edge_attr_emb.reset_parameters()
        self.att.reset_parameters()
        self.lin.reset_parameters()


    def forward(self, x: Tensor, edge_index: Adj, 
                edge_attr: OptTensor = None,
                size: Size = None) -> Tensor:

        x = self.lin_x(x)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        if self.concat:
            if self.root_weight:
                x = x.view(-1, 1, self.out_channels)
                out += x
            out = out.view(-1, self.heads * self.out_channels)
           
        else:
            out = out.mean(dim=1)
            if self.root_weight:
                out += x

        return out

    def message(self, x_i: Tensor, x_j: Tensor, 
                edge_attr: Tensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        edge_attr = F.leaky_relu(self.edge_attr_emb(edge_attr),
                                 self.negative_slope)
      
        alpha = torch.cat([x_i, x_j, edge_attr], dim=-1)
        alpha = F.leaky_relu(self.att(alpha), self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)

        out = self.lin(torch.cat([x_j, edge_attr], dim=-1)).unsqueeze(-2) 
        out = out * alpha.unsqueeze(-1)
       
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
        
        

class EdgeConv(MessagePassing):
    """
    Edge Conv operation
    """
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="max")  
        self.mlp = Sequential(
            Linear(2 * in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        return self.propagate(edge_index, x=x)  # shape [num_nodes, out_channels]

    def message(self, x_j: Tensor, x_i: Tensor) -> Tensor:
        # x_j: Source node features of shape [num_edges, in_channels]
        # x_i: Target node features of shape [num_edges, in_channels]
        edge_features = torch.cat([x_i, x_j - x_i], dim=-1)
        return self.mlp(edge_features)  # shape [num_edges, out_channels]

class MLP(torch.nn.Module):
    """
    MLP Block
    """
    def __init__(self,in_channels,out_channels,out_layer=False):
        super(MLP, self).__init__()     
        self.linear = Linear(in_channels,out_channels)
        self.out_layer = out_layer
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.linear(x)
        if self.out_layer:
            out = self.sigmoid(x)
        else:
            out = self.relu(x)
        return out


class TrafficRepresentationNet(torch.nn.Module):
    """
    Prediction(Reconstruction) Network
    Parameters:
        in_node: node input dim.
        in_edge: edge input dim.
        out_channels: Output dim.
        num_heads: number of mul. attention
        out_conv: encoder output dim.
    return:
        cls_out: Classification ouput
        reg_out: Regression output
        conv_cls_out：First componet of the encoder output
        conv_reg_out：Second componet of the encoder output
    """
    def __init__(self,in_node,in_edge,out_channels,num_heads,out_conv):
         super(TrafficRepresentationNet, self).__init__()     
         self.in_node = in_node   
         self.in_edge = in_edge
         self.mlp_node = MLP(in_node,16)
         self.mlp_edge = MLP(in_edge,64)
         self.mlp_emb = MLP(out_conv*num_heads,64)
         self.mlp_dec = MLP(64,out_channels,out_layer=True)

         self.conv1 = EdgeConv(16,64)
         self.conv2 = EdgeConv(64,128)
         self.conv3 = EGATConvs(in_channels=128, out_channels=out_conv,
                      edge_dim=64, edge_attr_emb_dim=128, heads=num_heads)
         
         self.conv4 = EGATConvs(in_channels=128, out_channels=out_conv,
                      edge_dim=64, edge_attr_emb_dim=128, heads=num_heads)
            
         self.batch_norm1 = torch.nn.BatchNorm1d(64) 
         self.batch_norm12 = torch.nn.BatchNorm1d(128)
         self.batch_norm2 = torch.nn.BatchNorm1d(out_conv*num_heads)
         self.batch_norm22 = torch.nn.BatchNorm1d(out_conv*num_heads)
         self.sigmoid = torch.nn.Sigmoid()
         self.relu = nn.ReLU(inplace=True)

    def forward(self, x, edge_index, edge_attr):
         x = x[:,[0,1]]*0.01 #using positions as node features & normalize the postion features
         edge_attr = edge_attr[:,[5,6,7]] #using distance, sin(a), cos(a) as edge features
         edge_attr[:,0] = edge_attr[:,0]*0.01

         node_emb = self.mlp_node(x)
         edge_emb = self.mlp_edge(edge_attr)  
         conv_node = self.conv1(node_emb,edge_index) 
         conv_node = self.relu(self.batch_norm1(conv_node))   
    
         conv_node2 = self.conv2(conv_node,edge_index) 
         conv_node2 = self.relu(self.batch_norm12(conv_node2)) 
            
         conv_cls_out = self.conv3(conv_node2, edge_index, edge_emb)         
         conv_cls_out = self.relu(self.batch_norm2(conv_cls_out)) 
         
         emb_cls = self.mlp_emb(conv_cls_out)
         cls_out = self.mlp_dec(emb_cls)

         conv_reg_out = self.conv4(conv_node2, edge_index, edge_emb)         
         conv_reg_out = self.relu(self.batch_norm22(conv_reg_out))  
         
         emb_reg = self.mlp_emb(conv_reg_out)
         reg_out = self.mlp_dec(emb_reg)

         return cls_out, reg_out, conv_cls_out, conv_reg_out