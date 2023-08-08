import torch
import torch.nn as nn
import numpy as np


class FieldWeightedFactorizationMachineModel(nn.Module):
    __use_tensors_field_interact_calc = True

    def __init__(self, num_features, embed_dim, num_fields):
        super(FieldWeightedFactorizationMachineModel, self).__init__()

        # num_features -- number of different values over all samples, num_fields -- number of columns
        self.num_features = np.max(num_features) + 1  # number of entries for embedding = (max possible ind.) +1 to since indexing starting from 0
        self.embedding_dim = embed_dim
        self.num_fields = num_fields                  # length of X

        self.w0 = nn.Parameter(torch.zeros(1))  # w0 global bias
        self.bias = nn.Embedding(self.num_features, 1)  # biases w: for every field 1 dimension embedding

        self.embeddings = nn.Embedding(self.num_features, embed_dim)

        self.field_inter_weights = nn.Parameter(torch.zeros(num_fields, num_fields))

        with torch.no_grad():
            nn.init.trunc_normal_(self.bias.weight, std=0.01)
            nn.init.trunc_normal_(self.embeddings.weight, std=0.01)
            nn.init.trunc_normal_(self.field_inter_weights, std=0.01)

    def get_field_inter_weight(self, i, j):
        return self.field_inter_weights[i][j].item()

    def _get_fixed_field_inter_weights(self):
        return torch.sub(self.field_inter_weights, torch.diagflat(torch.diag(self.field_inter_weights)))

    def calc_factorization_interactions(self, emb):
        if self.__use_tensors_field_interact_calc:
            return self._calc_factorization_interactions_tensors(emb)
        else:
            return self._calc_factorization_interactions_nested_loops(emb)

    def _calc_factorization_interactions_nested_loops(self, emb):
        factorization_interactions = 0.0
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                # print(i,j)
                # print(i, emb[:, i, :])
                # print(j, emb[:, j, :])
                inner_prod = torch.sum(emb[:, i, :] * emb[:, j, :], dim=-1)
                # print(inner_prod)
                factorization_interactions += self.get_field_inter_weight(i, j) * inner_prod

        return factorization_interactions

    def _calc_factorization_interactions_tensors(self, emb):                                                # emb = (batch_size, num_fields, embedding_dim)
        field_inter_weights_fixed = self._get_fixed_field_inter_weights()                                   # (num_fields, num_fields)
        emb_mul_emb_T = torch.matmul(emb, torch.transpose(emb, 1, 2))
        inner_product = (emb_mul_emb_T * field_inter_weights_fixed).sum(1).sum(1)         # torch.inner  does not work; .sum(1).sum(1)  to stay with (batch_size, 1) shape
        return torch.mul(inner_product, 0.5)   # (batch_size, 1)

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
        bias_x = self.bias(x).squeeze()
        biases_sum = bias_x.squeeze().sum(1) if bias_x.dim() > 1 else bias_x.squeeze().sum(0)  # (batch_size, 1)

        factorization_interactions = self.calc_factorization_interactions(emb)  # (batch_size, 1)

        # print(self.w0, biases_sum, factorization_interactions)

        # Combine field interactions and factorization interactions
        output = self.w0 + biases_sum + factorization_interactions
        # output = torch.sigmoid(x.squeeze(output))
        return output


# # Pairwise interactions
# # instead of calc. \sum_i \sum_j <v_i, v_j> x_i x_j  we calc. 0.5 * [(\sum_i v_i x_i))^2 - (\sum_i (v_i x_i)^2)]
# square_of_sum = torch.pow(torch.sum(emb, dim=1), 2)
# sum_of_square = torch.sum(torch.pow(emb, 2), dim=1)
# pairwise_interactions = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)  # (batch_size, 1)
