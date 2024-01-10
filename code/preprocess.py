from load_data import *
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, EarlyStoppingCallback


MODEL_NAME = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def label_to_num(label):
  num_label = []
  with open('./code/dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

# load dataset
train_dataset = load_data("./data/dataset/train/train_sample.csv")
# dev_dataset = load_data("./data/dataset/train/dev_sample.csv") # validation용 데이터는 따로 만드셔야 합니다.
print(train_dataset.head())

train_label = label_to_num(train_dataset['label'].values)
# print(train_label)
# dev_label = label_to_num(dev_dataset['label'].values)

# tokenizing dataset
tokenized_train = tokenized_dataset(train_dataset, tokenizer)
# print(tokenized_train)
# tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

# make dataset for pytorch.
RE_train_dataset = RE_Dataset(tokenized_train, train_label)
# print(RE_train_dataset)
# RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)