# -*- coding: utf8 -*-
"""
======================================
    Project Name: RE-For-NER
    File Name: data_loader
    Author: czh
    Create Date: 2121/3/25
--------------------------------------
    Change Activity: 
======================================
"""
import os
import json
import copy
import logging

import torch
from torch.utils.data import TensorDataset

from ReNER.utils.utils import DataProcessor, get_entities

logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, title, text_a, objects):
        self.guid = guid
        self.title = title
        self.text_a = text_a
        self.objects = objects

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeature(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, input_len, segment_ids, subject_index,
                 object_start_ids, object_end_ids, objects):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.input_len = input_len
        self.object_start_ids = object_start_ids
        self.subject_index = subject_index
        self.object_end_ids = object_end_ids
        self.objects = objects

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class NerDataProcessor(DataProcessor):
    def __init__(self):
        super(NerDataProcessor, self).__init__()

    def get_train_examples(self, data_dir):
        return self.__create_examples(self._read_json(os.path.join(data_dir, 'train.json')), 'train')

    def get_dev_examples(self, data_dir):
        return self.__create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        return self.__create_examples(self._read_json(os.path.join(data_dir, "test.json")), 'test')

    def get_labels(self):
        return ["O", "product"]

    @staticmethod
    def __create_examples(lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            title = line['title']
            text_a = line['words']
            labels = line['labels']
            entities = get_entities(labels, id2label=None, markup='bios')
            examples.append(InputExample(guid=guid, title=title, text_a=text_a, objects=entities))
        return examples


def tokenize(text, vocab, do_lower_case=True):
    _tokens = []
    for c in text:
        if do_lower_case:
            c = c.lower()
        if c in vocab:
            _tokens.append(c)
        else:
            _tokens.append('[UNK]')
    return _tokens


def convert_examples_to_features(examples, tokenizer, max_seq_length, label2id, cls_token_at_end=False,
                                 cls_token="[CLS]", cls_token_segment_id=1, sep_token="[SEP]", pad_on_left=False,
                                 pad_token=0, pad_token_segment_id=0, sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    features = []
    for ex_index, example in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        text_a = example.text_a
        title = example.title

        tokens = tokenize(text_a, tokenizer.vocab)
        # tokens = tokenizer.tokenize(text_a)
        objects = example.objects
        start_ids = [[0] for _ in range(len(tokens))]
        end_ids = [[0] for _ in range(len(tokens))]
        subject_index = []
        for i in range(len(tokens)):
            if i < len(title):
                subject_index.append(1)
            else:
                subject_index.append(0)
        object_id = []
        for object_ in objects:
            label = object_[0]
            start = object_[1]
            end = object_[2]

            start_ids[start][0] = 1
            end_ids[end][0] = 1
            object_id.append([label2id[label], start, end])

        # Account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            start_ids = start_ids[: (max_seq_length - special_tokens_count)]
            end_ids = end_ids[: (max_seq_length - special_tokens_count)]
            subject_index = subject_index[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        subject_index += [0]
        start_ids += [[0]]
        end_ids += [[0]]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        if cls_token_at_end:
            tokens += [cls_token]
            start_ids += [[0]]
            end_ids += [[0]]
            segment_ids += [cls_token_segment_id]
            subject_index += [0]
        else:
            tokens = [cls_token] + tokens
            start_ids = [[0]] + start_ids
            end_ids = [[0]] + end_ids
            segment_ids = [cls_token_segment_id] + segment_ids
            subject_index = [0] + subject_index

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_len = len(input_ids)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([1 if mask_padding_with_zero else 0] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            start_ids = ([[0]] * padding_length) + start_ids
            end_ids = ([[0]] * padding_length) + end_ids
            subject_index = [0] * padding_length + subject_index
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [1 if mask_padding_with_zero else 0] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            start_ids += ([[0]] * padding_length)
            end_ids += ([[0]] * padding_length)
            subject_index += [0] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(start_ids) == max_seq_length
        assert len(end_ids) == max_seq_length
        assert len(subject_index) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("subject_index: %s", " ".join([str(x) for x in subject_index]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("start_ids: %s" % " ".join([str(x) for x in start_ids]))
            logger.info("end_ids: %s" % " ".join([str(x) for x in end_ids]))

        features.append(InputFeature(
            input_ids=input_ids,
            input_mask=input_mask,
            input_len=input_len,
            segment_ids=segment_ids,
            subject_index=subject_index,
            object_start_ids=start_ids,
            object_end_ids=end_ids,
            objects=object_id
        ))
    return features


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_input_mask, all_subject_index, all_segment_ids, all_object_start_ids, \
        all_object_end_ids, all_lens = map(torch.stack, zip(*batch))

    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_input_mask = all_input_mask[:, :max_len]
    all_subject_index = all_subject_index[:, :max_len]
    all_segment_ids = all_segment_ids[:, :max_len]
    all_object_start_ids = all_object_start_ids[:, :max_len]
    all_object_end_ids = all_object_end_ids[:, :max_len]
    return all_input_ids, all_input_mask, all_subject_index, all_segment_ids, all_object_start_ids, \
        all_object_end_ids, all_lens


def load_and_cache_examples(args, tokenizer, processor, data_type='train'):
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_span-{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.train_max_seq_length if data_type == 'train' else args.eval_max_seq_length)))

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        label2id = {label: i for i, label in enumerate(label_list)}
        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                label2id=label2id,
                                                max_seq_length=args.train_max_seq_length
                                                if data_type == 'train' else args.eval_max_seq_length,
                                                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                sep_token=tokenizer.sep_token,
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    # Convert to Tensors and build dataset
    if data_type in ['dev', 'test']:
        return features
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_subject_index = torch.tensor([f.subject_index for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_object_start_ids = torch.tensor([f.object_start_ids for f in features], dtype=torch.long)
    all_object_end_ids = torch.tensor([f.object_end_ids for f in features], dtype=torch.long)
    all_input_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_subject_index, all_segment_ids, all_object_start_ids,
                            all_object_end_ids, all_input_lens)
    return dataset
