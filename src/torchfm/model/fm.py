import torch

from src.torchfm.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear


class FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim, is_multivalued=False):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim, is_multival=is_multivalued)
        self.linear = FeaturesLinear(field_dims, is_multival=is_multivalued)
        self.fm = FactorizationMachine(reduce_sum=True)

    # def get_l2_reg(self, emb):
    #     return [self.embedding.get_l2_reg(emb) + self.fm.get_l2_reg(emb), self.linear.get_l2_reg(emb)]

    def forward(self, x, return_l2=False):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        :param return_l2
        """
        emb, reg_emb = self.embedding(x, return_l2)
        x1, reg_rest = self.linear(x, return_l2)
        x2 = self.fm(emb)
        x = x1 + x2
        score = x.squeeze(1)

        if return_l2:
            return score, [reg_emb, reg_rest]       #torch.sigmoid()  - remove sigmoid since train/test with bcewithlogit
        else:
            return score, [0.0, 0.0]
