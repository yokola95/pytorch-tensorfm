import torch

from src.torchfm.layer import FeaturesEmbedding, CrossNetwork, MultiLayerPerceptron, FeaturesLinear


class DeepCrossNetworkModel(torch.nn.Module):
    """
    A pytorch implementation of Deep & Cross Network.

    Reference:
        R Wang, et al. Deep & Cross Network for Ad Click Predictions, 2017.
    """

    def __init__(self, field_dims, embed_dim, num_layers, mlp_dims, dropout):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cn = CrossNetwork(self.embed_output_dim, num_layers)
        #self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        # self.linear = torch.nn.Linear(self.embed_output_dim, 1)
        self.linear = FeaturesLinear(field_dims, is_multival=is_multivalued)
        #self.linear = torch.nn.Linear(mlp_dims[-1] + self.embed_output_dim, 1)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x, reg_emb = self.embedding(x)
        embed_x = embed_x.view(-1, self.embed_output_dim)
        x_l1, reg_cn = self.cn(embed_x)
        #h_l2 = self.mlp(embed_x)
        #x_stack = torch.cat([x_l1, h_l2], dim=1)
        x_stack = torch.cat([x_l1], dim=1)
        p, reg_lin = self.linear(x_stack)
        return torch.sigmoid(p.squeeze(1)), [reg_emb,reg_cn + reg_lin]
