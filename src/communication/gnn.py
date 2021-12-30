import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse

class GCNComm(torch.nn.Module):
    def __init__(self, input_shape, args, training=True):
        super(GCNComm, self).__init__()
        self.args = args
        self.training = training
        self.convs = []
        self.convs.append(GCNConv(input_shape, self.args.msg_hidden_dim))
        for i in range(1,self.args.num_layers-1):
            self.convs.append(GCNConv(self.args.msg_hidden_dim, self.args.msg_hidden_dim))
        self.convs.append(GCNConv(self.args.msg_hidden_dim, self.args.msg_out_size))    
    
    def cuda_transfer(self):
        for i in range(self.args.num_layers):
            self.convs[i].cuda()

    def forward(self,x, adj_matrix):
        x = []
        for x_in, am_in in torch.unbind(x, dim=0), torch.unbind(adj_matrix, dim=0):
            for i in range(self.args.num_layers):
                x_in = self.convs[i](x_in, dense_to_sparse(am_in)[0])
                if (i+1)<self.args.num_layers:
                    x_in = F.elu(x_in)
                    x_in = F.dropout(x_in, p=0.2, training=self.training)
            x.append(x_in) 
        return torch.stack(x, dim=0)
    
class GATComm(torch.nn.Module):
    def __init__(self, input_shape, args, training=True):
        super(GATComm, self).__init__()
        self.args = args

        self.convs = []
        self.convs.append(GCNConv(input_shape, self.args.msg_hidden_dim))
        for i in range(1,self.args.num_layers-1):
            self.convs.append(GCNConv(self.args.msg_hidden_dim, self.args.msg_hidden_dim))
        self.convs.append(GCNConv(self.args.msg_hidden_dim, self.args.msg_out_size))    
    
    def forward(self,x, adj_matrix):
        for i in range(self.args.num_layers):
            x = self.convs[i](x, edge_index)
            if (i+1)<self.args.num_layers:
                x = F.elu(x)
                x = F.dropout(x, p=0.2, training=self.training) 
        return x 