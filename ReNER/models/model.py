# -*- coding: utf8 -*-
"""
======================================
    Project Name: RE-For-NER
    File Name: model
    Author: czh
    Create Date: 2021/3/23
--------------------------------------
    Change Activity: 
======================================
"""
# 将ner问题建模为关系抽取问题，已知subject_entity和relation，预测object entity。
# 一个subject对应多个object。


import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCELoss
from transformers import BertModel, BertPreTrainedModel

from ReNER.layers.linear import PoolerStartLogits, PoolerEndLogits, PoolerEndLogitsOnStart
from ReNER.losses.focal_loss import FocalLoss
from ReNER.losses.label_smoothing import LabelSmoothingCrossEntropy


class Dense(nn.Module):
    """
        dense层
    """

    def __init__(self, input_size, out_size, activation="relu", dropout_rate=0.5):
        super(Dense, self).__init__()
        self.linear_layer = nn.Linear(input_size, out_size)
        self.dropout = nn.Dropout(dropout_rate)
        if activation == "sigmoid":
            self.active_layer = nn.Sigmoid()
        else:
            self.active_layer = nn.ReLU()

    def forward(self, input_tensor):
        linear_result = self.linear_layer(input_tensor)
        return self.active_layer(linear_result)


class MyModel(BertPreTrainedModel):
    def __init__(self, config, args, hidden_size=768):
        super(MyModel, self).__init__(config)
        self.soft_label = args.soft_label
        self.hidden_size = hidden_size
        self.num_labels = 1
        self.bert = BertModel(config)
        self.o_start_fc = PoolerStartLogits(self.hidden_size, self.num_labels)

        # 预测o_end时是否考虑o_start的预测结果
        if self.soft_label:
            self.o_end_fc = PoolerEndLogitsOnStart(self.hidden_size+self.num_labels, self.num_labels)
        else:
            self.o_end_fc = PoolerEndLogits(self.hidden_size, self.num_labels)
        self.pred_obj_heads = Dense(self.hidden_size, self.num_labels, activation='sigmoid',
                                    dropout_rate=args.dropout_rate)
        self.pred_obj_tails = Dense(self.hidden_size, self.num_labels, activation='sigmoid',
                                    dropout_rate=args.dropout_rate)
        self.drop_out = nn.Dropout(0.5)
        self.activation = nn.Sigmoid()

        # self.classifier = Classifier(self.hidden_size, use_deep=args.deep)

        self.loss_type = args.loss_type
        if self.loss_type == 'lsr':
            self.loss_func = LabelSmoothingCrossEntropy()
            # self.loss_func = LabelSmoothing()
        elif self.loss_type == 'foc':
            self.loss_func = FocalLoss()
        elif self.loss_type == 'ce':
            self.loss_func = CrossEntropyLoss()
        elif self.loss_type == 'bce':
            self.loss_func = BCELoss()
        else:
            raise ValueError('loss type must be [ce, foc, lsr]')

        self.init_weights()

    @staticmethod
    def seg_gather(x, idxs):
        batch_size, seq_len, hidden_size = x.size()
        assert list(idxs.size()) == [batch_size, 2]

        s_embed = []
        for batch_id in range(batch_size):
            idx = idxs[batch_id]
            start = idx[0].item()
            end = idx[1].item()
            embed = x[batch_id, start: end+1, :]
            s_embed.append(torch.mean(embed, dim=0).unsqueeze(0))
        s_embed = torch.cat(s_embed).to(x.device)
        return s_embed

    def cal_loss(self, pred, gold, mask):
        pred = pred.to(torch.float)
        gold = gold.to(torch.float)
        loss = self.loss_func(pred, gold)
        if loss.shape != mask.shape:
            mask = mask.unsqueeze(-1)
        loss = torch.sum(loss * mask) / torch.sum(mask)
        return loss

    def forward(self, input_ids, input_mask, s_index, o_start=None, o_end=None, token_type_ids=None):
        """
        :param input_ids: [batch_size, max_seq_len]
        :param s_index: [batch_size, max_seq_len]
        :param o_start: [batch_size, max_seq_len, 1]
        :param o_end: [batch_size, max_seq_len, 1]
        :param input_mask: [batch_size, max_seq_len]  the real length of sequences
        :param token_type_ids: [batch_size, max_seq_len]
        :return:
        """
        out = self.bert(input_ids=input_ids, token_type_ids=token_type_ids)
        seq_out = out[0]  # [batch_size, max_seq_len, hidden_dim]
        # s_embed = self.seg_gather(seq_out, s_index)  # [batch_size, hidden_size]
        if s_index.shape != seq_out.shape:
            s_index = s_index.unsqueeze(-1)
        s_embed = torch.mean(torch.mul(s_index, seq_out), dim=1, keepdim=True)
        # s_feature = s_embed.unsqueeze(1)
        token_features = seq_out + s_embed
        o_start_logits = self.pred_obj_heads(token_features)
        o_end_logits = self.pred_obj_tails(token_features)

        if o_start is None and o_end is None:
            return o_start_logits, o_end_logits

        obj_start_loss = self.cal_loss(o_start_logits, o_start, input_mask)
        obj_end_loss = self.cal_loss(o_end_logits, o_end, input_mask)
        total_loss = obj_start_loss + obj_end_loss
        return total_loss, o_start_logits, o_end_logits
