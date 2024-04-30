import argparse
from modelling_gpt_neox_moe import GPTNeoXForCausalLM
from transformers import GPTNeoXConfig
import json
from safetensors.torch import load_model
import transformers
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, load_from_disk
import evaluate
import torch
from eval import *
from transformers.utils import logging
import logging as logger

logging.set_verbosity_info()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main(args):
  with open(f"{args.weight_path}/config.json", "r") as fp:
    config = GPTNeoXConfig(**json.load(fp))
  model = GPTNeoXForCausalLM(config)
  tokenizer = transformers.GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
  tokenizer.add_special_tokens({'pad_token': '[PAD]'})
  load_model(model, f"{args.weight_path}/model-00001-of-00001.safetensors", strict=False)
  logger.info("model loaded")
  if args.dataset == "arxiv":
    dataset = arxiv()
  elif args.dataset == "uspto":
    dataset = uspto()
  elif args.dataset == "freelaw":
    dataset = freelaw()
  elif args.dataset == "github":
    dataset = github()
  logger.info("dataset loaded")
  metric = evaluate.load("accuracy")

  def tokenizer_func(x):
    return tokenizer(x["text"])
  
  def compute_metrics(eval_preds):
      preds, labels = eval_preds
      return metric.compute(predictions=preds, references=labels)
  
  def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
      logits = logits[0]
    return logits.argmax(dim=-1)


  tokenized_dataset = dataset.as_streaming_dataset(split="validation").map(tokenizer_func, batched=True)
  trainer = Trainer(
    model,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    args=TrainingArguments(
      "/mnt/scratch/bborisov/models/gpt_neox_moe/", 
      eval_accumulation_steps=1, 
      per_device_eval_batch_size=1
    ),
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
  )
  logger.info("trainer_loaded")
  torch.set_grad_enabled(False),
  metrics = trainer.evaluate()
  print(metrics)
  trainer.save_metrics("/mnt/scratch/bborisov/models/gpt_neox_moe/eval_res.json")
  trainer.log_metrics()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
        "--dataset",
        type=str,
        help="The dataset to use for training, must be a path to a jsonl file.",
  )
  parser.add_argument(
    "--weight_path",
    type=str
  )
  args = parser.parse_args()
  main(args)