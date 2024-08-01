import torch

from src.torchfm.model.weighted_embedding_bag import CompatibleWeightedEmbeddingBag, CompatibleEmbedding


class EmbeddingFactory:
    @staticmethod
    def get_embedding(num_embed, emb_dim, sparse, is_multival=False):
        return CompatibleEmbedding(num_embed, emb_dim,
                                   sparse=sparse) if not is_multival else CompatibleWeightedEmbeddingBag(16, 11,
                                                                                                         num_embed,
                                                                                                         emb_dim,
                                                                                                         device=None,
                                                                                                         dtype=torch.float32,
                                                                                                         sparse=sparse)
