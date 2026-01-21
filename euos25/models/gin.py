import torch
import torch_geometric
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import MLP, global_add_pool


class GIN(torch.nn.Module):
    def __init__(self, hidden_dim=64, output_dim=1, num_layers=5, dropout_p=0.2):
        super().__init__()
        # fields used for computing node embedding
        self.node_encoder = AtomEncoder(hidden_dim)

        self.convs = torch.nn.ModuleList(
            [
                torch_geometric.nn.conv.GINConv(
                    MLP([hidden_dim, hidden_dim, hidden_dim])
                )
                for _ in range(0, num_layers)
            ]
        )
        self.bns = torch.nn.ModuleList(
            [
                torch.nn.BatchNorm1d(num_features=hidden_dim)
                for _ in range(0, num_layers - 1)
            ]
        )
        self.dropout_p = dropout_p
        # end fields used for computing node embedding
        # fields for graph embedding
        self.pool = global_add_pool
        self.linear_hidden = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear_out = torch.nn.Linear(hidden_dim, output_dim)
        # end fields for graph embedding

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.linear_hidden.reset_parameters()
        self.linear_out.reset_parameters()

    def forward(self, batched_data):
        x, edge_index, batch = (
            batched_data.x,
            batched_data.edge_index,
            batched_data.batch,
        )
        # compute node embedding
        x = self.node_encoder(x)
        for idx in range(0, len(self.convs)):
            x = self.convs[idx](x, edge_index)
            if idx < len(self.convs) - 1:
                x = self.bns[idx](x)
                x = torch.nn.functional.relu(x)
                x = torch.nn.functional.dropout(
                    x, self.dropout_p, training=self.training
                )
        # note x is raw logits, NOT softmax'd
        # end computation of node embedding
        # convert node embedding to a graph level embedding using pooling
        x = self.pool(x, batch)
        x = torch.nn.functional.dropout(x, self.dropout_p, training=self.training)
        # transform the graph embedding to the output dimension
        # MLP after graph embed ensures we are not requiring the raw pooled node embeddings to be linearly separable
        x = self.linear_hidden(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, self.dropout_p, training=self.training)
        out = self.linear_out(x)
        return out
