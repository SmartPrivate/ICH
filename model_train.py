import transformers.training_args
from transformers import DataCollatorWithPadding, BertForSequenceClassification, BertTokenizer
from datasets import Dataset
import pandas
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import TrainingArguments, Trainer

df = pandas.read_csv('result/data_set.csv', sep='\t')
train_set = Dataset.from_pandas(df=df[['text', 'label']])
train_set = train_set.train_test_split(test_size=0.1, seed=42)

checkpoint = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(checkpoint)
model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=4)


def process_sample(sample):
    processed_text = sample['text']
    tokenized_text = tokenizer(processed_text, truncation=True, padding='max_length', max_length=256)
    return tokenized_text


train_set = train_set.map(process_sample)


def compute_metrics(pred):
    labels = pred.label_ids
    predicts = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicts, average='macro')
    result = classification_report(labels, predicts, target_names=['1', '2', '3', '4'], output_dict=False)
    with open('classification_report.txt', 'a', encoding='utf-8') as f:
        f.write(result)
    acc = accuracy_score(labels, predicts)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


training_args = TrainingArguments(
    output_dir='./train_results',  # output directory
    evaluation_strategy=transformers.training_args.IntervalStrategy.EPOCH
)

trainer = Trainer(
    model=model,  # the instantiated   Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_set['train'],  # training dataset
    eval_dataset=train_set['test'],  # evaluation dataset
    compute_metrics=compute_metrics
)

trainer.train()

model.save_pretrained('ich_model')
tokenizer.save_pretrained('ich_tokenizer')

