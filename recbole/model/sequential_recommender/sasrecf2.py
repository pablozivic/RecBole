# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:32
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

r"""
SASRecF
################################################
"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, FeatureSeqEmbLayer
from recbole.model.loss import BPRLoss
from recbole.utils import FeatureType


class SASRecF2(SequentialRecommender):
    """This is an extension of SASRec, which concatenates item representations and item attribute representations
    as the input to the model.
    """

    def __init__(self, config, dataset):
        super(SASRecF2, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.selected_features = config["selected_features"]
        self.pooling_mode = config["pooling_mode"]
        self.device = config["device"]

        for field in self.selected_features:
            if field not in dataset.field2type:
                raise RuntimeError(
                    f'Invalid field {field} in selected_features! Available: {dataset.field2type.keys()}'
                )

        self.num_feature_field = sum(
            1
            if dataset.field2type[field] != FeatureType.FLOAT_SEQ
            else dataset.num(field)
            for field in config["selected_features"]
        )

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        # define layers and loss
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.feature_embed_layer = FeatureSeqEmbLayer(
            dataset,
            self.hidden_size,
            self.selected_features,
            self.pooling_mode,
            self.device,
        )

        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.adapter_layer = nn.Linear(
            self.hidden_size * self.num_feature_field, self.hidden_size
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "COS":
            self.loss_fct = nn.CosineEmbeddingLoss(reduction='mean')
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['COS']!")

        # parameters initialization
        self.apply(self._init_weights)
        self.other_parameter_name = ["feature_embed_layer"]

        self._item_features = None
        self._item_features_version = None
        self.item_catalog = dataset.item_feat

    @property
    def item_features_table(self):
        current_version = self.feature_embed_layer.checksum()
        if self._item_features_version is None or self._item_features_version != current_version:
            # TODO: batch
            self._item_features = self.feature_embed_layer(None, self.item_catalog)
            self._item_features_version = current_version
        return self._item_features

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        # position embedding
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = self.embed_items(item_seq)

        input_emb = input_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        seq_output = self.gather_indexes(output, item_seq_len - 1)
        return seq_output  # [B H]

    def embed_items(self, item_seq):
        sparse_embedding, dense_embedding = self.feature_embed_layer(None, item_seq)
        sparse_embedding = sparse_embedding["item"]
        dense_embedding = dense_embedding["item"]
        # concat the sparse embedding and float embedding
        feature_table = []
        if sparse_embedding is not None:
            feature_table.append(sparse_embedding)
        if dense_embedding is not None:
            feature_table.append(dense_embedding)
        feature_table = torch.cat(feature_table, dim=-2)
        table_shape = feature_table.shape
        feat_num, embedding_size = table_shape[-2], table_shape[-1]
        feature_emb = feature_table.view(
            table_shape[:-2] + (feat_num * embedding_size,)
        )
        return self.adapter_layer(feature_emb)

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "COS":
            import ipdb;ipdb.set_trace()
            pos_items_emb = self.embed_items(pos_items)
            loss = self.loss_fct(seq_output, pos_items_emb)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        import ipdb;ipdb.set_trace()
        test_item_emb = self.item_features_table[test_item]
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        import ipdb;ipdb.set_trace()
        test_items_emb = self.item_features_table
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, item_num]
        return scores
