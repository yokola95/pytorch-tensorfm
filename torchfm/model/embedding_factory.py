import torch.nn as nn
from torchfm.model.weighted_embedding_bag import CompatibleWeightedEmbeddingBag


class EmbeddingFactory:
    @staticmethod
    def get_embedding(num_embed, emb_dim, sparse, is_multival=False):
        return nn.Embedding(num_embed, emb_dim, sparse=sparse) if not is_multival else CompatibleWeightedEmbeddingBag(16, 11, num_embed, emb_dim, sparse=sparse)
