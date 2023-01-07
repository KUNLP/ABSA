import argparse
import os
import json

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from utils import set_seed, collate_fn
from data import preprocessor
from model import ABSAModel

def train(args, model, data_reader, train_features, dev_features, test_features):
    train_dataloader = DataLoader(train_features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn,
                                  drop_last=True)
    total_steps = int(len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    new_layer = ["extractor"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": 5e-4},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    print('Total steps: {}'.format(total_steps))
    print('Warmup steps: {}'.format(warmup_steps))

    num_steps = 0
    best_score, best_epoch = 0, 0

    for epoch in range(int(args.num_train_epochs)):
        model.zero_grad()
        for step, batch in enumerate(tqdm(train_dataloader, desc="epoch "+str(epoch))):
            model.train()
            inputs = {'input_ids': batch[0].to(args.device),
                      'attention_masks': batch[1].to(args.device),
                      'category_targets': batch[2].to(args.device),
                      }
            loss = model(**inputs)
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if step % args.gradient_accumulation_steps == 0:
                num_steps += 1
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            if step % 5 == 0:
                print("loss:", loss.item())

        score = evaluate(args, model, dev_features, data_reader)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            if args.save_dir != "":
                if not os.path.exists(args.save_dir):
                    os.mkdir(args.save_dir)
            torch.save(model.state_dict(), os.path.join(args.save_dir, "final_model.pt"))
            report(args, model, test_features, data_reader)

        print("Best_score:", best_score, "// epoch:", best_epoch)

def evaluate(args, model, features, data_reader, tag='dev'):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, collate_fn=collate_fn, drop_last=False)
    predicts = []

    for i_b, batch in enumerate(dataloader):
        model.eval()
        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_masks': batch[1].to(args.device),
                  }
        with torch.no_grad():
            batch_pred = model(**inputs)
            predicts += batch_pred.cpu().numpy().tolist()

    ca_targets = [f["category_target"] for f in features]
    se_targets = [f["sentiment_target"] for f in features]

    r_cnt, p_cnt, c_cnt = 0, 0, 0
    pair_c_cnt = 0
    for i, (target, predict) in enumerate(zip(ca_targets, predicts)):
        r_cnt += sum(target)
        p_cnt += sum(predict)
        c_cnt += sum([a and b for a, b in zip(target, predict)])

        target_pair = make_pair(data_reader, target, se_targets[i])
        predict_pair = make_pair(data_reader, predict, [0, 0, 0, 0, 0, 0])

        for t in target_pair:
            if t in predict_pair:
                pair_c_cnt += 1

    eps = 1e-10
    recall = c_cnt / r_cnt
    precision = c_cnt / p_cnt
    f1 = (recall * precision * 2) / (recall + precision + eps)

    pair_recall = pair_c_cnt / r_cnt
    pair_precision = pair_c_cnt / p_cnt
    pair_f1 = (pair_recall * pair_precision * 2) / (pair_recall + pair_precision + eps)

    print("========= Performance =========")
    print("\tRecall: ", round(recall, 5))
    print("\tPrecision: ", round(precision, 5))
    print("\tF1:", round(f1, 5))
    print("\tpair F1:", round(pair_f1, 5))
    print()

    return f1


def error_analysis(args, model, features, data_reader, tag='dev'):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, collate_fn=collate_fn, drop_last=False)
    predicts = []

    for i_b, batch in enumerate(dataloader):
        model.eval()
        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_masks': batch[1].to(args.device),
                  }
        with torch.no_grad():
            batch_pred = model(**inputs)
            predicts += batch_pred.cpu().numpy().tolist()

    ca_targets = [f["category_target"] for f in features]
    se_targets = [f["sentiment_target"] for f in features]

    r_cnt, p_cnt, c_cnt = 0, 0, 0
    pair_c_cnt = 0
    for i, (target, predict) in enumerate(zip(ca_targets, predicts)):
        r_cnt += sum(target)
        p_cnt += sum(predict)
        c_cnt += sum([a and b for a, b in zip(target, predict)])

        target_pair = make_pair(data_reader, target, se_targets[i])
        predict_pair = make_pair(data_reader, predict, [0, 0, 0, 0, 0, 0])

        for t in target_pair:
            if t in predict_pair:
                pair_c_cnt += 1

    id2ca = data_reader.id2category

    for i, _ in enumerate(features):
        target_id2ca = []
        for cid, val in enumerate(ca_targets[i]):
            if val == 1:
                target_id2ca.append(id2ca[cid])
        target_id2ca.sort()

        predict_id2ca = []
        for cid, val in enumerate(predicts[i]):
            if val == 1:
                predict_id2ca.append(id2ca[cid])
        predict_id2ca.sort()

        if target_id2ca != predict_id2ca:
            print(features[i]["sentence"])
            print("target", target_id2ca)
            print("predict", predict_id2ca)
            print()

    eps = 1e-10
    recall = c_cnt / r_cnt
    precision = c_cnt / p_cnt
    f1 = (recall * precision * 2) / (recall + precision + eps)

    pair_recall = pair_c_cnt / r_cnt
    pair_precision = pair_c_cnt / p_cnt
    pair_f1 = (pair_recall * pair_precision * 2) / (pair_recall + pair_precision + eps)

    print("========= Performance =========")
    print("\tRecall: ", round(recall, 5))
    print("\tPrecision: ", round(precision, 5))
    print("\tF1:", round(f1, 5))
    print("\tpair F1:", round(pair_f1, 5))
    print()


def make_pair(data_reader, binary_ca, se):
    pairs = []
    se_idx = 0

    id2category = data_reader.id2category
    id2sentiment = data_reader.id2sentiment
    for ca_id, prop in enumerate(binary_ca):
        if prop == 1:
            pairs.append([id2category[ca_id], id2sentiment[se[se_idx]]])
            se_idx += 1

    return pairs

def report(args, model, features, data_reader, tag='test'):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, collate_fn=collate_fn, drop_last=False)
    predicts = []

    for i_b, batch in enumerate(dataloader):
        model.eval()
        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_masks': batch[1].to(args.device),
                  }
        with torch.no_grad():
            batch_pred = model(**inputs)
            predicts += batch_pred.cpu().numpy().tolist()

    with open(args.result_dir, encoding="utf-8", mode="w") as wf:
        for i, data in enumerate(features):
            d_id = data["id"]
            sent = data["sentence"]
            pred_category = predicts[i]
            pred_sentiment = [0 for _ in range(10)]
            pairs = make_pair(data_reader, pred_category, pred_sentiment)

            result = {"id": d_id, "sentence_form":sent, "annotation": pairs}
            wf.write(json.dumps(result, ensure_ascii=False)+"\n")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_flag", default=0, type=bool)
    parser.add_argument("--data_dir", default="data/", type=str)
    parser.add_argument("--model_name_or_path", default="klue/roberta-large", type=str)
    parser.add_argument("--cache_dir", default="cache/", type=str)

    parser.add_argument("--save_dir", default="models/", type=str)
    parser.add_argument("--load_dir", default="models/", type=str)
    parser.add_argument("--result_dir", default="results/result.json", type=str)

    parser.add_argument("--train_file", default="nikluge-sa-2022-train.jsonl", type=str)
    parser.add_argument("--dev_file", default="nikluge-sa-2022-dev.jsonl", type=str)
    parser.add_argument("--test_file", default="nikluge-sa-2022-test.jsonl", type=str)

    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=64, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass.")
    parser.add_argument("--warmup_ratio", default=0.05, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=33,
                        help="random seed for initialization")
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--category_size", type=int, default=25)

    args = parser.parse_args()

    if not os.path.exists(args.cache_dir):
        os.mkdir(args.cache_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    if args.seed > 0:
        set_seed(args)

    config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    config.gradient_checkpointing = True

    data_reader = preprocessor.Preprocessor()
    config.gradient_checkpointing = True

    train_features = data_reader.preprocessing_v2(os.path.join(args.data_dir, args.train_file))
    dev_features = data_reader.preprocessing_v2(os.path.join(args.data_dir, args.dev_file))
    test_features = data_reader.preprocessing_v2(os.path.join(args.data_dir, args.test_file))

    model = ABSAModel(args, config)
    model.to(0)

    if args.train_flag == 0:
        train(args, model, data_reader, train_features, dev_features, test_features)
    elif args.train_flag == 2:
        model.load_state_dict(torch.load(os.path.join(args.load_dir, "final_model.pt")))
        error_analysis(args, model, dev_features, data_reader)


if __name__ == "__main__":
    main()
