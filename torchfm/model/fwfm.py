from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import numpy as np


class BaseFieldWeightedFactorizationMachineModel(nn.Module):
    def __init__(self, num_features, embed_dim, num_fields):
        super(BaseFieldWeightedFactorizationMachineModel, self).__init__()

        # num_features -- number of different values over all samples, num_fields -- number of columns
        self.num_features = np.max(num_features) + 1  # number of entries for embedding = (max possible ind.) +1 to since indexing starting from 0
        self.embedding_dim = embed_dim
        self.num_fields = num_fields                    # length of X

        self.w0 = nn.Parameter(torch.zeros(1))          # w0 global bias
        self.bias = nn.Embedding(self.num_features, 1)  # biases w: for every field 1 dimension embedding (num_features, 1)

        self.embeddings = nn.Embedding(self.num_features, embed_dim)  # embedding vectors V: (num_features, embedding_dim)

        with torch.no_grad():
            nn.init.trunc_normal_(self.bias.weight, std=0.01)
            nn.init.trunc_normal_(self.embeddings.weight, std=0.01)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields)``
        """
        # Embedding layer
        emb = self.embeddings(x)  # (batch_size, num_fields, embedding_dim)

        # Biases (field weights)
        biases_sum = self.bias(x).squeeze().sum(-1)   # (batch_size, 1)    # TODO: check squeeze on dim -1 or -2??

        factorization_interactions = self.calc_factorization_interactions(emb)  # (batch_size, 1)

        # Combine field interactions and factorization interactions
        output = self.w0 + biases_sum + factorization_interactions
        return output             # (batch_size, 1)

    @abstractmethod
    def calc_factorization_interactions(self, emb):
        pass


class Symmetric(nn.Module):
    def forward(self, x):
        return x.triu(1) + x.triu(1).transpose(-1, -2)   # 0 diagonal, no need to remove diagonal in the code


class FieldWeightedFactorizationMachineModel(BaseFieldWeightedFactorizationMachineModel):
    __use_tensors_field_interact_calc = True

    def __init__(self, num_features, embed_dim, num_fields):
        super(FieldWeightedFactorizationMachineModel, self).__init__(num_features, embed_dim, num_fields)
        self.field_inter_weights = self._init_interaction_weights(num_fields)
        parametrize.register_parametrization(self, "field_inter_weights", Symmetric())      # ???

    def _init_interaction_weights(self, num_fields):
        aux = torch.empty(num_fields, num_fields)
        with torch.no_grad():
            nn.init.trunc_normal_(aux, std=0.01)   # aux = aux + aux.transpose(0, 1)    # .triu() + aux.triu(1).transpose(-1, -2)  # make it symmetric           # TODO: check parametrization
        return nn.Parameter(aux)

    def calc_factorization_interactions(self, emb):              # emb = (batch_size, num_fields, embedding_dim)
        if self.__use_tensors_field_interact_calc:
            return self._calc_factorization_interactions_tensors(emb)
        else:
            return self._calc_factorization_interactions_nested_loops(emb)

    def _get_field_inter_weight(self, i, j):
        return self.field_inter_weights[i][j]

    def _get_fixed_field_inter_weights(self):  ### todo: remove???
        return torch.sub(self.field_inter_weights, torch.diagflat(torch.diag(self.field_inter_weights)))

    def _calc_factorization_interactions_nested_loops(self, emb):
        factorization_interactions = 0.0
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                inner_prod = torch.sum(emb[..., i, :] * emb[..., j, :], dim=-1)
                factorization_interactions += self._get_field_inter_weight(i, j) * inner_prod

        return factorization_interactions

    def _calc_factorization_interactions_tensors(self, emb):                                                # emb = (batch_size, num_fields, embedding_dim)
        field_inter_weights_fixed = self._get_fixed_field_inter_weights()                                   # (num_fields, num_fields)
        emb_mul_emb_T = torch.matmul(emb, torch.transpose(emb, -2, -1))
        inner_product = (emb_mul_emb_T * field_inter_weights_fixed).sum([-1, -2])   # torch.inner  does not work; .sum(1).sum(1)  to stay with (batch_size, 1) shape
        return inner_product / 2   # (batch_size, 1)



# # Pairwise interactions
# # instead of calc. \sum_i \sum_j <v_i, v_j> x_i x_j  we calc. 0.5 * [(\sum_i v_i x_i))^2 - (\sum_i (v_i x_i)^2)]
# square_of_sum = torch.pow(torch.sum(emb, dim=1), 2)
# sum_of_square = torch.sum(torch.pow(emb, 2), dim=1)
# pairwise_interactions = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)  # (batch_size, 1)
