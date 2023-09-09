from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import numpy as np

from torchfm.torch_utils.constants import sparseGrads


class BaseFieldWeightedFactorizationMachineModel(nn.Module):
    def __init__(self, num_features, embed_dim, num_fields):
        super(BaseFieldWeightedFactorizationMachineModel, self).__init__()

        # num_features -- number of different values over all samples, num_fields -- number of columns
        self.num_features = np.max(num_features) + 1  # number of entries for embedding = (max possible ind.) +1 to since indexing starting from 0
        self.embedding_dim = embed_dim
        self.num_fields = num_fields                    # length of X

        self.w0 = nn.Parameter(torch.zeros(1))          # w0 global bias
        self.bias = nn.Embedding(self.num_features, 1, sparse=sparseGrads)  # biases w: for every field 1 dimension embedding (num_features, 1)

        self.embeddings = nn.Embedding(self.num_features, embed_dim, sparse=sparseGrads)  # embedding vectors V: (num_features, embedding_dim)

        with torch.no_grad():
            nn.init.trunc_normal_(self.bias.weight, std=0.01)
            nn.init.trunc_normal_(self.embeddings.weight, std=0.01)

    def calc_linear_term(self, x):
        # Biases (field weights)
        biases_sum = self.bias(x).squeeze().sum(-1)   # (batch_size, 1)

        return self.w0 + biases_sum                   # (batch_size, 1)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields)``
        """
        # Embedding layer
        emb = self.embeddings(x)  # (batch_size, num_fields, embedding_dim)

        # linear term = global bias and biases per feature (field weights)
        lin_term = self.calc_linear_term(x)   # (batch_size, 1)

        factorization_interactions = self.calc_factorization_interactions(emb)  # (batch_size, 1)

        # Combine field interactions and factorization interactions
        output = lin_term + factorization_interactions
        return output             # (batch_size, 1)

    @abstractmethod
    def calc_factorization_interactions(self, emb):
        pass


class Symmetric(nn.Module):
    def forward(self, x):
        return x.triu(1) + x.triu(1).transpose(-1, -2)   # zero diagonal and symmetric - due to parametrization no need to remove diagonal in the code


class FWFM_topk:
    _topk = -1
    _topk_vals = None
    _topk_indices = None
    _fwfm_instance = None

    def __init__(self, fwfm):
        self._fwfm_instance = fwfm

    @property
    def topk(self):
        return self._topk

    @topk.setter
    def topk(self, k):
        self._topk = k

    def set_top_entries_from_field_inter_weights(self):
        if self.topk <= 0:
            self._topk_vals = None
            self._topk_indices = None
            return

        input_tensor = torch.abs(self._fwfm_instance.field_inter_weights.triu(1).abs())    # 2D tensor
        flat_tensor = input_tensor.view(-1)
        topk_values, topk_indices_flat = torch.topk(flat_tensor, self.topk)  # from 1D tensor

        # Convert the flat indices back to 2D indices
        topk_indices_2d = torch.tensor([divmod(idx.item(), input_tensor.shape[1]) for idx in topk_indices_flat])
        # The original values
        topk_values_original = input_tensor[topk_indices_2d[:, 0], topk_indices_2d[:, 1]]

        self._topk_vals = topk_values_original
        self._topk_indices = topk_indices_2d

    def calc_factorization_interactions(self, emb):
        if self.topk <= 0:
            return 0.0

        factorization_interactions = 0.0
        for val, ind in zip(self._topk_vals, self._topk_indices):
            i = ind[0]
            j = ind[1]
            inner_prod = torch.sum(emb[..., i, :] * emb[..., j, :], dim=-1)
            factorization_interactions += val * inner_prod

        return factorization_interactions


class FieldWeightedFactorizationMachineModel(BaseFieldWeightedFactorizationMachineModel):
    __use_tensors_field_interact_calc = True
    _topk_obj = None
    _use_topk = False

    def __init__(self, num_features, embed_dim, num_fields, topk):
        super(FieldWeightedFactorizationMachineModel, self).__init__(num_features, embed_dim, num_fields)
        self.field_inter_weights = self._init_interaction_weights(num_fields)
        self._topk_obj = FWFM_topk(self)
        parametrize.register_parametrization(self, "field_inter_weights", Symmetric())
        self.topk = topk

    @property
    def use_topk(self):
        return self._use_topk

    @use_topk.setter
    def use_topk(self, use):
        self._use_topk = use
        if self._use_topk:
            self._topk_obj.set_top_entries_from_field_inter_weights()

    @property
    def topk(self):
        return self._topk_obj.topk

    @topk.setter
    def topk(self, k):
        self._topk_obj.topk = k

    def calc_factorization_interactions(self, emb):              # emb = (batch_size, num_fields, embedding_dim)
        if self.use_topk:
            return self._topk_obj.calc_factorization_interactions(emb)
        elif self.__use_tensors_field_interact_calc:
            return self._calc_factorization_interactions_tensors(emb)
        else:
            return self._calc_factorization_interactions_nested_loops(emb)

    def _init_interaction_weights(self, num_fields):
        aux = torch.empty(num_fields, num_fields)
        with torch.no_grad():
            nn.init.trunc_normal_(aux, std=0.01)
        return nn.Parameter(aux)

    def _get_field_inter_weight(self, i, j):
        return self.field_inter_weights[i][j]

    # Mostly for debugging (comparison for an equality of the output)
    def _calc_factorization_interactions_nested_loops(self, emb):
        factorization_interactions = 0.0
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                inner_prod = torch.sum(emb[..., i, :] * emb[..., j, :], dim=-1)
                factorization_interactions += self._get_field_inter_weight(i, j) * inner_prod

        return factorization_interactions

    def _calc_factorization_interactions_tensors(self, emb):                       # emb = (batch_size, num_fields, embedding_dim)
        emb_mul_emb_T = torch.matmul(emb, torch.transpose(emb, -2, -1))
        inner_product = (emb_mul_emb_T * self.field_inter_weights).sum([-1, -2])   # inner_product = (batch_size, 1)   # due to parametrization, self.field_inter_weights is zero-diagonal and symmetric matrix of size (num_fields, num_fields)
        return inner_product / 2   # (batch_size, 1)

