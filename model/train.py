from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import datasets
import torch

# Load the dataset
dataset = datasets.load_dataset('glue', 'sst2')

# Subsample 10 examples from the training and validation sets
dataset['train'] = dataset['train']
dataset['validation'] = dataset['validation']

# Load the tokenizer and encode the dataset
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
encoded_dataset = dataset.map(lambda example: tokenizer(example['sentence'], padding='max_length', truncation=True), batched=True)

# Load the model and configure the training arguments
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy='epoch',
    load_best_model_at_end=True,
)

# Define the function to compute the accuracy
def compute_accuracy(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {'accuracy': (predictions == labels).mean()}

# Define the data collator to stack tensors across the batch dimension
def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(example['input_ids']) for example in batch])
    attention_mask = torch.stack([torch.tensor(example['attention_mask']) for example in batch])
    labels = torch.tensor([example['label'] for example in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# Define the trainer and fine-tune the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['validation'],
    data_collator=collate_fn,    # use the collate_fn to collate the input data
    compute_metrics=compute_accuracy,
)
trainer.train()

tokenizer.save_pretrained('model/saved_model/tokenizer')
model.save_pretrained('model/saved_model/distilbert')