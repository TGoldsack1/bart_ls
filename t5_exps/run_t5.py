from transformers import T5Tokenizer, T5ForConditionalGeneration, EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_metric, load_dataset, Dataset, DatasetDict
import torch
import ast, re, json
import numpy as np
import pandas as pd
import nltk


# Set variables
# "bigbird-pegasus-large-pubmed" "facebook/bart-large-cnn" "Kevincp560/bart-large-finetuned-pubmed" "facebook/bart-large" 
# "mse30/bart-base-finetuned-pubmed", "facebook/bart-base"



TRAIN = True
ds = "PLOS"


# data_dir = f"/home/acp20tg/bart_ls/resources/{ds}_fs-controllable_all"
data_dir = f"/home/acp20tg/bart_ls/resources/{ds}_fs"
output_dir = f"/home/acp20tg/bart_ls/results/{ds}/controllable/t5"

max_output_len = 425 if ds == "eLife" else 275
min_output_len = 275 if ds == "eLife" else 125

metric = load_metric("rouge")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = T5Tokenizer.from_pretrained("t5-base",  model_max_length=max_output_len)
model = T5ForConditionalGeneration.from_pretrained("t5-base", max_length=max_output_len, min_length=min_output_len).to(device)



print(device)
print(data_dir.split("/")[2]) 
print(min_output_len)
print(max_output_len)

def load_data(ddir, split):
  fp_src = ddir + "/" + split + ".src"
  fp_tgt = ddir + "/" + split + ".tgt"

  with open(fp_src, "r") as in_f:
    src_data = in_f.readlines()

  with open(fp_tgt, "r") as in_f:
    tgt_data = in_f.readlines()

  return src_data, tgt_data

def batch_tokenize_fn(examples, max_source_length=1024, max_target_length=1024):
  """
  Generate the input_ids and labels field for huggingface dataset/dataset dict.
  
  Truncation is enabled, so we cap the sentence to the max length, padding will be done later
  in a data collator, so pad examples to the longest length in the batch and not the whole dataset.
  """
  sources = examples["document"]
  targets = examples["summary"]

  model_inputs = tokenizer(sources, max_length=max_source_length, truncation=True)

  # setup the tokenizer for targets,
  # huggingface expects the target tokenized ids to be stored in the labels field
  with tokenizer.as_target_tokenizer():
    labels = tokenizer(targets, max_length=max_target_length, truncation=True)

  model_inputs["labels"] = labels["input_ids"]
  return model_inputs

  
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

if TRAIN: 

  dataset_dict = DatasetDict()

  for split in ["train", "val", "test"]:
    docs, sums = load_data(data_dir, split)
    data_dict = {}
    # docs = [x['abstract'] + flatten(x['sections']) for x in data]
    # sums = [x['summary'] for x in data]

    d = {'docs': docs, 'sums': sums} 
    test_df = pd.DataFrame(d)
    test_df = test_df.dropna()


    data_dict["document"] = test_df["docs"].tolist()
    data_dict["summary"] = test_df["sums"].tolist()
    data_dict = Dataset.from_dict(data_dict)
    tokenized_data_dict = data_dict.map(batch_tokenize_fn, batched=True, \
      fn_kwargs={"max_source_length": 1024, "max_target_length": 1024 })

    dataset_dict[split] = tokenized_data_dict

  # Need to include num epochs and rouge metric 
  training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir, 
    logging_dir=output_dir,
    logging_steps=100,         
    evaluation_strategy="epoch",
    eval_accumulation_steps = 20,
    num_train_epochs = 50,
    log_level = "info",
    save_strategy="epoch",
    seed=123,
    predict_with_generate=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_total_limit=2,
    load_best_model_at_end=True,
    fp16=True,
    metric_for_best_model="rouge2"
  )

  callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
  data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

  trainer = Seq2SeqTrainer(
    model=model, 
    args=training_args, 
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["val"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=callbacks
  )

  trainer.train()
