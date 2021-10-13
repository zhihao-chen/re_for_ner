# -*- coding: utf8 -*-
"""
======================================
    Project Name: RE-For-NER
    File Name: kl_loss
    Author: czh
    Create Date: 2021/8/4
--------------------------------------
    Change Activity: 
======================================
"""
import torch.nn.functional as func


def compute_kl_loss(p, q, pad_mask=None, merge_mode="sum"):
    p_loss = func.kl_div(func.log_softmax(p, dim=-1), func.softmax(q, dim=-1), reduction='none')
    q_loss = func.kl_div(func.log_softmax(q, dim=-1), func.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        if pad_mask.shape != p.shape:
            pad_mask = pad_mask.unsqueeze(-1)
        p_loss = p_loss * pad_mask
        q_loss = q_loss * pad_mask

    # You can choose whether to use function "sum" and "mean" depending on your task
    if merge_mode == "sum":
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()
    elif merge_mode == "mean":
        p_loss = p_loss.mean()
        q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss

