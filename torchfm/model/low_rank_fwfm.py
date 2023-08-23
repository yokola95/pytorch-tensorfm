import torch
import torch.nn as nn
from torch.linalg import multi_dot
from torchfm.model.fwfm import BaseFieldWeightedFactorizationMachineModel


class LowRankFieldWeightedFactorizationMachineModel(BaseFieldWeightedFactorizationMachineModel):
    def __init__(self, num_features, embed_dim, num_fields, c):
        super(LowRankFieldWeightedFactorizationMachineModel, self).__init__(num_features, embed_dim, num_fields)

        self.diag_e = nn.Parameter(torch.empty(c))  # (c)
        self.U = nn.Parameter(torch.empty(c, num_fields))  # (c, num_fields)

        with torch.no_grad():
            nn.init.trunc_normal_(self.diag_e, std=0.01)
            nn.init.trunc_normal_(self.U, std=0.01)

    def _calc_diag_d(self):  # the efficient way
        tmp = self.U.t() * self.diag_e
        return -(self.U * tmp.t()).sum(0)       # (num_fields)

    def _calc_diag_d_straighforward(self):
        interaction_matrix = multi_dot([self.U.t(), torch.diag(self.diag_e), self.U])  # R_tilda = (num_fields, num_fields)  <-  (c, num_fields)^T x (c, c) x (c, num_fields)
        return -torch.diag(interaction_matrix)  # (num_fields)

    def calc_factorization_interactions(self, emb):  # emb = (batch_size, num_fields, embedding_dim)
        """
        Calculates low rank factorization interactions
        """
        P = torch.matmul(self.U, emb)                                                  # (batch_size, c, embedding_dim)
        diag_d = self._calc_diag_d()                                                   # (num_fields)

        term1 = (torch.pow(emb, 2).sum(-1) * diag_d).sum(1)   # .sum(-1)                            # (batch_size) <- (batch_size, num_fields) * (num_fields).sum(1)
        term2 = (torch.pow(P, 2).sum(-1) * self.diag_e).sum(1)                         # (batch_size) <- (batch_size, c) * (c).sum(1)
        return (term1 + term2) / 2                                                     # (batch_size, 1)
