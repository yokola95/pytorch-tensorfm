# Copyright 2023, Yahoo.

import torch
import torch.nn as nn
import torch.nn.functional as f


class WeightedEmbeddingBag(nn.Module):
    r"""
    A class similar to ``nn.EmbeddingBag`` which supports mini-batches, assuming that the number of bags
    in all samples in a mini-batch is the same.

    :param num_embeddings: the number of emebdding vectors to hold
    :type num_embeddings int

    :param embedding_dim: the dimension of each embedding vector
    :type embedding_dim int

    :param emb_kwargs: parameter dict for the :class:`nn.Embedding` constructor.
    """
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None, _freeze=False, **emb_kwargs):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs),
                                    requires_grad=not _freeze)
        self.emb_kwargs = emb_kwargs
        torch.nn.init.normal_(self.weight)

    def get_l2_reg(self, embeddings):
        return embeddings.square().mean(0).sum()

    def forward(self, input, offsets, per_sample_weights, return_l2=False):
        r"""
        Computed weighted sums of input embeddings in each bag, assuming each mini-batch comprises the same
        number of embeddings, weights, and bags. Variable number of embeddings and their corresponding weights
        per sample is possible with padding. However, the number of bags per sample has to be equal for all
        mini-batch samples. Returns a tensor of weighted-sums of embedding vectors in each sample.

        :param input: BxN matrix, where each row contains per-sample embedding indices.
        :type input: torch.Tensor

        :param offsets: BxM offsets pointing to end-of-bag indices inside each sample. Note, that this differs from
                        torch.nn.EmbeddingBag, where offsets point to the start-of-bag indices.
        :type offsets: torch.Tensor

        :param per_sample_weights: BxN matrix, where each row contaisn per-sample embedding weights.
        :type per_sample_weights: torch.Tensor

        :return: BxM tensor of weighted sums of embedding bags.
        :rtype: torch.Tensor
        """
        embeddings = torch.nn.functional.embedding(input, self.weight, **self.emb_kwargs)
        weighted_embeddings = embeddings * per_sample_weights.unsqueeze(2)
        padded_summed = f.pad(weighted_embeddings, [0, 0, 1, 0, 0, 0]).cumsum(dim=1)
        padded_offsets = f.pad(offsets, [1, 0, 0, 0], value=-1) + 1

        def batch_gather(input, off):
            emb_dim = input.shape[2]
            batch_size = off.shape[0]
            num_offsets = off.shape[1]
            i = torch.arange(batch_size, device=input.device).reshape(batch_size, 1, 1)
            j = off.reshape(batch_size, num_offsets, 1)
            k = torch.arange(emb_dim, device=input.device)

            return input[i, j, k]

        score = batch_gather(padded_summed, padded_offsets[:, 1:]) - batch_gather(padded_summed, padded_offsets[:, :-1])
        reg = self.get_l2_reg(embeddings) if return_l2 else 0.0
        return score, reg


class CompatibleWeightedEmbeddingBag(WeightedEmbeddingBag):
    def __init__(self, in_l, o_l, num_embeddings, embedding_dim, device=None, dtype=None, _freeze=False, **emb_kwargs):
        super(CompatibleWeightedEmbeddingBag, self).__init__(num_embeddings, embedding_dim, device=device, dtype=dtype, _freeze=_freeze, **emb_kwargs)
        self.in_l = in_l   # indices length
        self.o_l = o_l     # offsets length

    def forward(self, gen_input, return_l2=False):
        input, offsets, per_sample_weights = gen_input[..., 0:self.in_l].long(), gen_input[..., self.in_l:(self.in_l + self.o_l)].long(), gen_input[..., (self.in_l+self.o_l):]
        return super(CompatibleWeightedEmbeddingBag, self).forward(input, offsets, per_sample_weights, return_l2)


class CompatibleEmbedding(nn.Embedding):

    def __init__(self, num_embed, emb_dim, sparse):
        super(CompatibleEmbedding, self).__init__(num_embed, emb_dim, sparse=sparse)

    def get_l2_reg(self, embeddings):
        return embeddings.square().mean(0).sum()

    def forward(self, input, return_l2=False):
        embeddings = super(CompatibleEmbedding, self).forward(input)
        reg = self.get_l2_reg(embeddings) if return_l2 else 0.0
        return embeddings, reg
