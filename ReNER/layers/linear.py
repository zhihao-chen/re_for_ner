# -*- coding: utf8 -*-
"""
======================================
    Project Name: RE-For-NER
    File Name: linear
    Author: czh
    Create Date: 2021/3/23
--------------------------------------
    Change Activity: 
======================================
"""
import torch.nn as nn


class PoolerStartLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerStartLogits, self).__init__()
        self.dense = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states):
        x = self.dense(hidden_states)
        return x


class PoolerEndLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerEndLogits, self).__init__()
        self.dense = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states):
        x = self.dense(hidden_states)
        return x


class PoolerEndLogitsOnStart(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerEndLogitsOnStart, self).__init__()
        self.dense_0 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dense_1 = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states):
        x = self.dense_0(hidden_states)
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x)
        return x


class Classifier(nn.Module):
    def __init__(self, hidden_size, num_classes=1, use_deep=False, drop_rate=0.5):
        super(Classifier, self).__init__()

        self.use_deep = use_deep
        self.dense_0 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.layerNorm = nn.LayerNorm(hidden_size)
        self.dense_1 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(p=drop_rate)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, s_o_embed):
        if self.use_deep:
            x = self.dense_0(s_o_embed)
            x = self.activation(x)
            x = self.layerNorm(x)
            x = self.dense_1(x)
        else:
            x = self.dense_1(s_o_embed)
        x = self.dropout(x)
        pred_label = self.softmax(x)
        return pred_label
