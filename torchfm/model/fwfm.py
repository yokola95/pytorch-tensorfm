import torch
import torch.nn as nn
import numpy as np


class FieldWeightedFactorizationMachineModel(nn.Module):
    def __init__(self, num_fields, embed_dim, num_factors):
        super(FieldWeightedFactorizationMachineModel, self).__init__()

        self.num_fields = np.max(num_fields) + 1  # num indices (max possible ind.), +1 to start from 1 and not from 0
        self.embedding_dim = embed_dim
        self.num_factors = num_factors  # length of X

        self.w0 = nn.Parameter(torch.zeros(1))  # w0 global bias
        self.bias = nn.Embedding(self.num_fields, 1)  # biases w

        self.embeddings = nn.Embedding(self.num_fields, embed_dim)

        self.field_inter_weights = torch.zeros(num_factors, num_factors)

        with torch.no_grad():
            nn.init.trunc_normal_(self.bias.weight, std=0.01)
            nn.init.trunc_normal_(self.embeddings.weight, std=0.01)
            nn.init.trunc_normal_(self.field_inter_weights, std=0.01)

    def get_field_inter_weight(self, i, j):
        return self.field_inter_weights[i][j].item()

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields)``
        """
        # print(self.num_fields)
        # print(self.num_factors)
        # print(self.embeddings)
        # print(x)

        # Embedding layer
        emb = self.embeddings(x)  # (batch_size, num_fields, embedding_dim)

        # Biases (field weights)
        biases_sum = self.bias(x).squeeze().sum(1)

        # # Pairwise interactions
        # # instead of calc. \sum_i \sum_j <v_i, v_j> x_i x_j  we calc. 0.5 * [(\sum_i v_i x_i))^2 - (\sum_i (v_i x_i)^2)]
        # square_of_sum = torch.pow(torch.sum(emb, dim=1), 2)
        # sum_of_square = torch.sum(torch.pow(emb, 2), dim=1)
        # pairwise_interactions = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)  # (batch_size, 1)

        factorization_interactions = 0
        for i in range(self.num_factors - 1):
            for j in range(i + 1, self.num_factors):
                # print(i,j)
                # print(i, emb[:, i, :])
                # print(j, emb[:, j, :])
                inner_prod = torch.sum(emb[:, i, :] * emb[:, j, :], dim=-1)
                # print(inner_prod)
                factorization_interactions += self.get_field_inter_weight(i, j) * inner_prod
                # print(factorization_interactions)

        # factorization_interactions = torch.sum(torch.pow(torch.matmul(emb, self.field_weights.t()), 2), dim=1, keepdim=True)  # (batch_size, 1)

        # Combine field interactions and factorization interactions
        output = self.w0 + biases_sum + factorization_interactions  # + pairwise_interactions
        # output = torch.sigmoid(x.squeeze(output))
        return output
