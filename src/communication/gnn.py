import torch
from torch_geometric.nn import GCNConv

class GCNComm(torch.nn.Module):
    def __init__(self, input_shape, args):
        super(GCNComm, self).__init__()
        self.args = args

        self.convs = []
        self.convs.append(GCNConv(input_shape, self.args.msg_hidden_dim))
        for i in range(1,self.args.num_layers-1):
            self.convs.append(GCNConv(self.args.msg_hidden_dim, self.args.msg_hidden_dim))
        self.convs.append(GCNConv(self.args.msg_hidden_dim, self.args.msg_out_size))    
    
    def forward(self,x, edge_index):
        for i in range(self.args.num_layers):
            x = self.convs[i](x, edge_index)
            x = x.relu()        
    
class GATComm(torch.nn.Module):
    def __init__(self, input_shape, args):
        super(GATComm, self).__init__()
        self.args = args

        self.convs = []
        self.convs.append(GCNConv(input_shape, self.args.msg_hidden_dim))
        for i in range(1,self.args.num_layers-1):
            self.convs.append(GCNConv(self.args.msg_hidden_dim, self.args.msg_hidden_dim))
        self.convs.append(GCNConv(self.args.msg_hidden_dim, self.args.msg_out_size))    
    