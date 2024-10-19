import torch
# import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TFAutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset

dataset = load_dataset("imdb")
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_data(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

encoded_dataset = dataset.map(preprocess_data, batched=True)
# PyTorch
encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
train_dataset_pt = encoded_dataset['train']
test_dataset_pt = encoded_dataset['test']
model_pt = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model_pt,
    args=training_args,
    train_dataset=train_dataset_pt,
    eval_dataset=test_dataset_pt,
)

trainer.train()
trainer.evaluate(test_dataset_pt)