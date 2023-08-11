import torch
import torch.nn as nn
import numpy as np
from torch.linalg import multi_dot


class LowRankFieldWeightedFactorizationMachineModel(nn.Module):
    def __init__(self, num_features, embed_dim, num_fields, c):
        super(LowRankFieldWeightedFactorizationMachineModel, self).__init__()

        # num_features -- number of different values over all samples, num_fields -- number of columns
        self.num_features = np.max(num_features) + 1  # number of entries for embedding = (max possible ind.) +1 to since indexing starting from 0
        self.embedding_dim = embed_dim
        self.num_fields = num_fields                  # length of X
        self.c = c

        self.w0 = nn.Parameter(torch.empty(1))  # w0 global bias
        self.bias = nn.Embedding(self.num_features, 1)  # biases w: for every field 1 dimension embedding  (num_features, 1)

        self.embeddings = nn.Embedding(self.num_features, embed_dim)  # embedding vectors V: (num_features, embedding_dim)

        self.diag_e = nn.Parameter(torch.empty(c))                    # (c)

        self.U = nn.Parameter(torch.empty(c, num_fields))             # (c, num_fields)

        with torch.no_grad():
            nn.init.trunc_normal_(self.bias.weight, std=0.01)
            nn.init.trunc_normal_(self.embeddings.weight, std=0.01)
            nn.init.trunc_normal_(self.diag_e, std=0.01)
            nn.init.trunc_normal_(self.U, std=0.01)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields)``
        """
        # Embedding layer
        emb = self.embeddings(x)  # (batch_size, num_fields, embedding_dim)

        # Biases (field weights)
        bias_x = self.bias(x).squeeze()                                                        # (batch_size, num_fields)
        biases_sum = bias_x.squeeze().sum(1) if bias_x.dim() > 1 else bias_x.squeeze().sum(0)  # (batch_size, 1)

        factorization_interactions = self.calc_low_rank_factorization_interactions(emb)  # (batch_size, 1)

        # print(self.w0, biases_sum, factorization_interactions)

        # Combine field interactions and factorization interactions
        output = self.w0 + biases_sum + factorization_interactions
        return output

    def calc_low_rank_factorization_interactions(self, emb):         # emb = (batch_size, num_fields, embedding_dim)
        P = torch.matmul(self.U, emb)                                # (batch_size, c, embedding_dim)
        interaction_matrix = multi_dot([self.U.t(), torch.diag(self.diag_e), self.U])       # R_tilda = (num_fields, num_fields)  <-  (c, num_fields)^T x (c, c) x (c, num_fields)
        diag_d = -torch.diag(interaction_matrix)                                            # (num_fields)
        term1 = (torch.linalg.vector_norm(emb, ord=2, dim=2)**2 * diag_d).sum(1)            # (batch_size) <- (batch_size, num_fields) * (num_fields).sum(1)
        term2 = (torch.linalg.vector_norm(P, ord=2, dim=2)**2 * self.diag_e).sum(1)         # (batch_size) <- (batch_size, c) * (c).sum(1)
        return 0.5 * (term1 + term2)                                                        # (batch_size, 1)
