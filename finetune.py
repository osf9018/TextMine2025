"""
Script pour la spécialisation d'un encodeur-décodeur pré-entraîné pour la tâche d'extraction de relations

Utilisation : python finetune.py --data dataset --checkpoint mt0-xxl --rank 64 --n_epochs 2
"""
__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"
__license__ = "MIT"

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
import argparse
import json
import pandas as pd
from collections import defaultdict
import numpy as np
import os
import torch; torch.set_float32_matmul_precision("high")

parser = argparse.ArgumentParser(description='LoRA on encoder-decoder style LM for relation extraction')
parser.add_argument("-r", "--rank", type=int, default=64)
parser.add_argument("-d", "--data", type=str, default="dataset")
parser.add_argument("-s", "--scheduler", type=str, default="cosine")
parser.add_argument("-lr", "--learning_rate", type=float, default=5e-5)
parser.add_argument("-a", "--alpha_scale", type=int, default=1)
parser.add_argument("-bs", "--batch_size", type=int, default=16)
parser.add_argument("-ga", "--gradient_accumulation", type=int, default=1)
parser.add_argument("-c", "--checkpoint", type=str, default="mt0-xxl")
parser.add_argument("-n", "--n_epochs", type=int, default=2)
parser.add_argument("-m", "--modules", type=str, default="qv")
parser.add_argument("-do", "--dropout", type=float, default=0.05)
parser.add_argument("-mx", "--max_length", type=int, default=512)
parser.add_argument("-f", "--format", type=str, default="yes/no")
parser.add_argument("-od", "--out_dir", type=str, default="output")
args = parser.parse_args()
print(args)

########
# LoRA #
########
modules = ["q", "v"]
if args.modules == "all":
    modules = ["q", "k", "v", "o", "wi_0", "wi_1"]
lora_config = LoraConfig(
    lora_dropout=args.dropout,
    r=args.rank,
    lora_alpha=args.rank * args.alpha_scale,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
    target_modules=modules,
)

#########
# MODEL #
#########
model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint,
                                              torch_dtype=torch.bfloat16,
                                              device_map="auto")
model = get_peft_model(model, lora_config)
model.generation_config.max_new_tokens = 6
model.generation_config.min_new_tokens = 1
print(model.generation_config)
model.print_trainable_parameters()
tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, truncation_side="right")

########
# DATA #
########
path = args.data
dataset = load_dataset(path)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
def preprocess_function(examples):
    prompts = ['Does the relation {head_entity: ['+examples["head_entity"][i]+'], relation_type: '+examples["type"][i]+', tail_entity: ['+examples["tail_entity"][i]+']} exists in the following text: "'+examples["text"][i].replace('"', "")+'"?' for i in range(len(examples["answer"]))]
    answers = ["yes" if a == "true" else "no" for a in examples["answer"]]
    if "flan" in args.checkpoint:
        prompts = [prompt.replace("{", "(").replace("}", ")") for prompt in prompts] # le tokenizer T5 ignore les accolades
    return tokenizer(prompts, text_target=answers, max_length=args.max_length, truncation=True)
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)
print(tokenizer.decode(tokenized_dataset["train"]["input_ids"][0]), tokenizer.decode(tokenized_dataset["train"]["labels"][0]))

##############
# VALIDATION #
##############
if "no_val" not in args.data:
    uids = set()
    for i in range(len(dataset["validation"])):
        uids.add(dataset["validation"][i]["uid"])
    val_df = pd.read_csv("data/train.csv")
    val_df = val_df[val_df["id"].isin(uids)]
    val_df = val_df.set_index("id")
    val_df.relations = val_df.relations.apply(json.loads)
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=False)
        predictions = ["yes" in dec.lower() for dec in decoded_preds]
        dict_preds = {}
        for uid in val_df.index:
            dict_preds[uid] = []
        for data, pred in zip(dataset["validation"], predictions):
            uid = data["uid"]
            parts = data["relation"][1:-1].split(", ")
            relation = (int(parts[0]), parts[1].replace("'", ""), int(parts[2]))
            if pred:
                dict_preds[uid].append(relation)
        uids = []
        pred_relations = []
        for uid, relations in dict_preds.items():
            uids.append(uid)
            pred_relations.append(relations)
        pred_df = pd.DataFrame({"id": uids, "relations": pred_relations})
        pred_df.relations = pred_df.relations.apply(lambda l: json.dumps(l))
        f, ntp, nfp, nfn  = score(val_df, pred_df)
        return {"f1": f, "ntp": ntp, "nfp": nfp, "nfn": nfn}
    def score(val_df, submission):
        submission = submission.set_index("id")
        submission.relations = submission.relations.apply(json.loads)
        false_negatives, false_positives, true_positives = [], [], []
        for index, data in val_df.iterrows():
            test_rels = [tuple(rel) for rel in data["relations"]]
            predicted_relations =[tuple(rel) for rel  in  submission.loc[index]["relations"]]
            false_positives += list(set(predicted_relations).difference(test_rels))
            false_negatives += list(set(test_rels).difference(predicted_relations))
            true_positives += list(set(test_rels).intersection(predicted_relations))
        fn_rels = defaultdict(list)
        tp_rels = defaultdict(list)
        fp_rels = defaultdict(list)
        for relation in false_negatives:
            fn_rels[relation[1]].append(relation)
        for relation in false_positives:
            fp_rels[relation[1]].append(relation)
        for relation in true_positives:
            tp_rels[relation[1]].append(relation)
        f1s = dict()
        for predicate in set(tp_rels.keys()).union(fn_rels.keys()):
            precision = 0 if len(fp_rels[predicate]+tp_rels[predicate])==0 else len(tp_rels[predicate])/len(tp_rels[predicate]+fp_rels[predicate])
            recall = 0 if len(fn_rels[predicate]+tp_rels[predicate])==0 else len(tp_rels[predicate])/len(tp_rels[predicate]+fn_rels[predicate])
            f1s[predicate] = 0 if recall+precision==0 else 2*precision*recall/(precision+recall)
        macro_f1 = sum(f1s.values())/len(f1s)
        return macro_f1, len(true_positives), len(false_positives), len(false_negatives)

############
# TRAINING #
############
save_steps = (len(dataset["train"]) / (args.batch_size * args.gradient_accumulation)) // 12

chkpt_name = os.path.basename(args.checkpoint)
out_dir = f"{args.out_dir}/{chkpt_name}-{args.data}-r{args.rank}-{args.modules}"
training_arguments = Seq2SeqTrainingArguments(
    report_to=None,
    seed=0,
    output_dir=out_dir,
    num_train_epochs=args.n_epochs,
    eval_strategy="no" if "no_val" in args.data else "steps",
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size*4,
    gradient_accumulation_steps=args.gradient_accumulation,
    optim="paged_adamw_8bit",
    eval_steps=save_steps,
    save_steps=save_steps,
    logging_steps=50,
    learning_rate=args.learning_rate,
    weight_decay=0.1,
    fp16=False,
    bf16=True,
    max_grad_norm=1,
    warmup_ratio=0.05,
    group_by_length=True,
    lr_scheduler_type=args.scheduler,
    metric_for_best_model="f1",
    greater_is_better=True,
    bf16_full_eval=True,
    save_total_limit=2,
    load_best_model_at_end=False if "no_val" in args.data else True
)
trainer = Seq2SeqTrainer(
    model=model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=None if "no_val" in args.data else tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    args=training_arguments,
    preprocess_logits_for_metrics=None if "no_val" in args.data else preprocess_logits_for_metrics,
    compute_metrics=None if "no_val" in args.data else compute_metrics,
)

trainer.train()
