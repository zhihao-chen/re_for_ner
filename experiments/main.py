# -*- coding: utf8 -*-
"""
======================================
    Project Name: RE-For-NER
    File Name: main
    Author: czh
    Create Date: 2021/3/23
--------------------------------------
    Change Activity: 
======================================
"""
import os
import time
import json

import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import RandomSampler, DataLoader, DistributedSampler
from transformers import BertConfig, BertTokenizer, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
from accelerate import Accelerator

from ReNER.models.model import MyModel
from ReNER.utils.data_loader import NerDataProcessor, load_and_cache_examples, collate_fn
from ReNER.utils.utils import extract_items
from ReNER.metrics.metric import SpanEntityScore
from experiments.config import get_argparse
from ReNER.utils.common import init_logger, logger, json_to_text, seed_everything
from ReNER.callbacks.adversarial import FGM
from ReNER.callbacks.progressbar import ProgressBar

accelerator = Accelerator()


def init_model_config_tokenizer(args, num_label):
    bert_config = BertConfig.from_pretrained(args.model_name_or_path if not args.config_name else args.config_name,
                                             num_labels=num_label,
                                             cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path if not args.tokenizer_name else args.tokenizer_name,
                                              do_lower_case=args.do_lower_case,
                                              cache_dir=args.cache_dir if args.cache_dir else None)
    model = MyModel.from_pretrained(args.model_name_or_path,
                                    from_tf=bool(".ckpt" in args.model_name_or_path),
                                    config=bert_config,
                                    args=args, hidden_size=bert_config.hidden_size)
    return bert_config, tokenizer, model


def init_optimizer_scheduler(args, params, t_total):
    optimizer = AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay, eps=args.adam_epsilon)
    args.num_warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps,
                                                num_training_steps=t_total)
    return optimizer, scheduler


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]

    bert_parameters = model.bert.named_parameters()
    start_parameters = model.o_start_fc.named_parameters()
    end_parameters = model.o_end_fc.named_parameters()

    optimizer_grouped_parameters = [
        {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.learning_rate},
        {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
            , 'lr': args.learning_rate},

        {"params": [p for n, p in start_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': 0.001},
        {"params": [p for n, p in start_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
            , 'lr': 0.001},

        {"params": [p for n, p in end_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': 0.001},
        {"params": [p for n, p in end_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
            , 'lr': 0.001},
    ]

    optimizer, scheduler = init_optimizer_scheduler(args, optimizer_grouped_parameters, t_total)
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    scaler = GradScaler()

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size
                * args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
                )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    steps_trained_in_current_epoch = 0
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    summary = SummaryWriter(args.log_dir, comment='Linear')

    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
        # set global_step to global_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    if args.do_adv:
        fgm = FGM(model, emb_name=args.adv_name, epsilon=args.adv_epsilon)
    model.zero_grad()

    best_f1 = 0.0
    min_eval_loss = 0.0
    best_epoch = 0
    patience = 0

    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    for epoch in range(int(args.num_train_epochs)):
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for step, batch in enumerate(train_dataloader):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch_ = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch_[0],
                "input_mask": batch_[1],
                "s_index": batch_[2],
                "o_start": batch_[4],
                "o_end": batch_[5]
            }
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = batch_[3] if args.model_type in ["bert", "xlnet"] else None
            if args.fp16:
                with autocast():
                    outputs = model(**inputs)
                    loss = outputs[0]
            else:
                outputs = model(**inputs)
                loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                scaler.scale(loss).backward()
            elif args.accelerator:
                accelerator.backward(loss)
            else:
                loss.backward()
            if args.do_adv:
                fgm.attack()
                loss_adv = model(**inputs)[0]
                if args.n_gpu > 1:
                    loss_adv = loss_adv.mean()
                loss_adv.backward()
                fgm.restore()
            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()  # Update learning rate schedule
                #logger.info('learning rate: %s',' '.join([str(l) for l in list(scheduler.get_lr())]))
                model.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    logger.info("\n")
                    if args.local_rank == -1:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        result = evaluate(args, model, tokenizer)
                        f1 = result['f1']
                        eval_loss = result['loss']
                        summary.add_scalar('Train/loss', loss, global_step)
                        summary.add_scalar('Eval/loss', result['loss'], global_step)
                        # if f1 >= best_f1:
                        #     best_f1 = f1
                        if eval_loss <= min_eval_loss:
                            min_eval_loss = eval_loss
                            best_epoch = epoch
                            patience = 0
                            # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
                            # Create output directory if needed
                            if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
                                os.makedirs(args.output_dir)
                            logger.info("Saving model checkpoint to %s", args.output_dir)
                            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                            # They can then be reloaded using `from_pretrained()`
                            model_to_save = (
                                model.module if hasattr(model, "module") else model
                            )  # Take care of distributed/parallel training
                            model_to_save.save_pretrained(args.output_dir)
                            tokenizer.save_vocabulary(args.output_dir)
                            # Good practice: save your training arguments together with the trained model
                            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    tokenizer.save_vocabulary(output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)
        patience += 1
        if patience > args.patience:
            break

        logger.info("\n")
        if 'cuda' in str(args.device):
            with torch.cuda.device(f'cuda:{args.cuda}'):
                torch.cuda.empty_cache()
    return global_step, tr_loss / global_step, best_f1, best_epoch


def evaluate(args, model, tokenizer, prefix=""):
    processor = NerDataProcessor()

    metric = SpanEntityScore(args.id2label)
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    eval_features = load_and_cache_examples(args, tokenizer, processor, data_type='dev')
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_features))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(eval_features), desc="Evaluating")
    for step, f in enumerate(eval_features):
        input_lens = f.input_len
        input_ids = torch.tensor([f.input_ids[:input_lens]], dtype=torch.long).to(args.device)
        input_mask = torch.tensor([f.input_mask[:input_lens]], dtype=torch.long).to(args.device)
        subject_index = torch.tensor([f.subject_index], dtype=torch.long).to(args.device)
        segment_ids = torch.tensor([f.segment_ids[:input_lens]], dtype=torch.long).to(args.device)
        object_start_ids = torch.tensor([f.object_start_ids[:input_lens]], dtype=torch.long).to(args.device)
        object_end_ids = torch.tensor([f.object_end_ids[:input_lens]], dtype=torch.long).to(args.device)
        objects = f.objects
        model.eval()
        with torch.no_grad():
            inputs = {"input_ids": input_ids, "input_mask": input_mask, "s_index": subject_index,
                      "o_start": object_start_ids, "o_end": object_end_ids}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (segment_ids if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)

        tmp_eval_loss, start_logits, end_logits = outputs
        # R = bert_extract_item(start_logits, end_logits)
        R = extract_items(start_logits, end_logits)
        T = objects
        #if step % 1000 == 0:
        #    print('1',start_logits)
        #    print('2',end_logits)
        #    print('R:',R)
        #    print('T',T)
        metric.update(true_subject=T, pred_subject=R)
        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        pbar(step)

    logger.info("\n")
    eval_loss = eval_loss / nb_eval_steps
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss

    logger.info("***** Eval results %s *****", prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    logger.info("***** Entity results %s *****", prefix)

    for key in sorted(entity_info.keys()):
        logger.info("******* %s results ********" % key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        logger.info(info)
    return results


def predict():
    args = get_argparse()
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    init_logger(log_file=args.output_dir + f'/{args.model_type}-{time_}.log')

    device = torch.device('cpu')
    processor = NerDataProcessor()

    label_list = processor.get_labels()
    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)
    args.num_labels = num_labels

    metric = SpanEntityScore(args.id2label)

    tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    config = BertConfig.from_pretrained(args.output_dir, num_labels=num_labels)
    model = MyModel.from_pretrained(args.output_dir, config=config, args=args, hidden_size=config.hidden_size)
    model.to(device)

    output_file = os.path.join(args.output_dir, 'test_results.txt')
    test_features = load_and_cache_examples(args, tokenizer, processor, data_type='test')
    print(len(test_features))
    # Eval!
    logger.info("***** Running prediction *****")
    logger.info("  Num examples = %d", len(test_features))
    logger.info("  Batch size = %d", 1)

    results = []
    output_predict_file = os.path.join(args.output_dir, "test_predict.json")
    pbar = ProgressBar(n_total=len(test_features), desc="Predicting")
    for step, f in enumerate(test_features):
        input_lens = f.input_len
        input_ids = torch.tensor([f.input_ids[:input_lens]], dtype=torch.long).to(device)
        input_mask = torch.tensor([f.input_mask[:input_lens]], dtype=torch.long).to(device)
        subject_index = torch.tensor([f.subject_index], dtype=torch.long).to(device)
        segment_ids = torch.tensor([f.segment_ids[:input_lens]], dtype=torch.long).to(device)
        object_start_ids = torch.tensor([f.object_start_ids[:input_lens]], dtype=torch.long).to(device)
        object_end_ids = torch.tensor([f.object_end_ids[:input_lens]], dtype=torch.long).to(device)
        objects = f.objects
        model.eval()
        with torch.no_grad():
            inputs = {"input_ids": input_ids, "input_mask": input_mask, "s_index": subject_index,
                      "o_start": object_start_ids, "o_end": object_end_ids}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = segment_ids if args.model_type in ["bert", "xlnet"] else None
            outputs = model(**inputs)
        test_loss, start_logits, end_logits = outputs
        R = extract_items(start_logits, end_logits)
        if R:
            label_entities = [[args.id2label[x[0]], x[1], x[2]] for x in R]
        else:
            label_entities = []
        T = objects
        metric.update(true_subject=T, pred_subject=R)
        json_d = {}
        json_d['id'] = step
        json_d['entities'] = label_entities
        results.append(json_d)
        pbar(step)
    logger.info("\n")

    eval_info, entity_info = metric.result()
    result = {f'{key}': value for key, value in eval_info.items()}

    logger.info("***** Eval results *****")
    info = "-".join([f' {key}: {value:.4f} ' for key, value in result.items()])
    logger.info(info)

    with open(output_predict_file, "w") as writer:
        for record in results:
            writer.write(json.dumps(record) + '\n')

    test_text = []
    with open(os.path.join(args.data_dir, "test.json"), 'r') as fr:
        data = json.load(fr)
        for line in data:
            test_text.append(line)
    test_submit = []
    for x, y in zip(test_text, results):
        json_d = {}
        json_d['text'] = x['text']
        json_d['true_label'] = {"product": x['product']}
        json_d['pred_label'] = {}
        entities = y['entities']
        words = list(x['text'])
        if len(entities) != 0:
            for subject in entities:
                tag = subject[0]
                start = subject[1]
                end = subject[2]
                word = "".join(words[start:end + 1])
                if not word:
                    continue
                if tag in json_d['pred_label']:
                    if word in json_d['pred_label'][tag]:
                        json_d['pred_label'][tag][word].append([start, end])
                    else:
                        json_d['pred_label'][tag][word] = [[start, end]]
                else:
                    json_d['pred_label'][tag] = {}
                    json_d['pred_label'][tag][word] = [[start, end]]
        test_submit.append(json_d)
    json_to_text(output_file, test_submit)


def main():
    args = get_argparse()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    init_logger(log_file=args.output_dir + f'/{args.model_type}-{time_}.log')
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        # args.n_gpu = torch.cuda.device_count()
        args.n_gpu = 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = accelerator.device if args.accelerator else device
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16, )
    # Set seed
    seed_everything(args.seed)

    processor = NerDataProcessor()
    label_list = processor.get_labels()
    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)
    args.num_labels = num_labels
    args.model_type = args.model_type.lower()

    bert_config, tokenizer, model = init_model_config_tokenizer(args, num_labels)
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, processor, data_type='train')
        global_step, tr_loss, best_f1, best_epoch = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s, best f1 = %s, best epoch = %s",
                    global_step, tr_loss, best_f1, best_epoch)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = MyModel.from_pretrained(args.output_dir, config=bert_config, args=args, hidden_size=bert_config.hidden_size)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))
    if args.do_predict:
        predict()


if __name__ == "__main__":
    main()
    #predict()
