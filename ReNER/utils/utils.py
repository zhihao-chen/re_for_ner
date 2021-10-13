# -*- coding: utf8 -*-
"""
======================================
    Project Name: RE-For-NER
    File Name: utils
    Author: czh
    Create Date: 2021/3/26
--------------------------------------
    Change Activity: 
======================================
"""
import torch
import csv
import json


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_text(cls, input_file):
        lines = []
        with open(input_file, 'r') as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append({"words": words, "labels": labels})
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                lines.append({"words": words, "labels": labels})
        return lines

    @staticmethod
    def _read_json(input_file):
        lines = []
        with open(input_file, 'r') as f:
            datas = json.load(f)
            label = 'product'
            for line in datas:
                title = line['title']
                if len(title) < 2:
                    continue
                text = line['text']
                label_entities = line.get(label, None)
                words = list(text)
                labels = ['O'] * len(words)
                if label_entities is not None:
                    for sub_name, sub_index in label_entities.items():
                        for start_index, end_index in sub_index:
                            assert ''.join(words[start_index:end_index+1]) == sub_name
                            if start_index == end_index:
                                labels[start_index] = 'S-' + label
                            else:
                                labels[start_index] = 'B-' + label
                                labels[start_index+1:end_index+1] = ['I-'+label]*(len(sub_name)-1)
                lines.append({'title': title, "words": words, "labels": labels})
        return lines


def get_entity_bios(seq, id2label):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
        id2label:
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx  # noqa
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entity_bio(seq, id2label):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
        id2label:
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entities(seq, id2label, markup='bios'):
    """
    :param seq:
    :param id2label:
    :param markup:
    :return:
    """
    assert markup in ['bio', 'bios']
    if markup == 'bio':
        return get_entity_bio(seq, id2label)
    else:
        return get_entity_bios(seq, id2label)


def bert_extract_item(start_logits, end_logits):
    s = []
    start_pred = torch.argmax(start_logits, -1).cpu().numpy()[0][1:-1]
    end_pred = torch.argmax(end_logits, -1).cpu().numpy()[0][1:-1]
    for i, s_l in enumerate(start_pred):
        if s_l == 0:
            continue
        for j, e_l in enumerate(end_pred[i:]):
            if s_l == e_l:
                s.append((s_l, i, i + j))
                break
    return s


def extract_items(start_logits, end_logits, h_bar=0.5, t_bar=0.5):
    entities = []
    batch_size, max_len, _ = start_logits.size()
    start_pred_idx = torch.where(start_logits[0] > h_bar)[0]
    end_pred_idx = torch.where(end_logits[0] > t_bar)[0]

    for s_index in start_pred_idx:
        e_indexs = end_pred_idx[end_pred_idx > s_index]
        if len(e_indexs) > 0:
            for e_index in e_indexs:
                e_index = e_index.item()
                if e_index == 0 or e_index == max_len-1:
                    continue
                else:
                    entities.append([1, s_index.item()-1, e_index-1])
                    break
    return entities


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
