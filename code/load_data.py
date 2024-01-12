import pickle as pickle
import os
import pandas as pd
import torch
import re


class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

# def preprocessing_dataset(dataset):
#   """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
#   subject_entity = []
#   object_entity = []
#   subject_type= []
#   object_type= []
#   for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
#     # i = i[1:-1].split(',')[0].split(':')[1]
#     sub_entity = eval(i)
#     ob_entity = eval(j)
#     sub_word= sub_entity['word']
#     sub_type = sub_entity['type']
#     # j = j[1:-1].split(',')[0].split(':')[1]
#     ob_word = ob_entity['word']
#     ob_type = ob_entity['type']


#     subject_entity.append(sub_word)
#     object_entity.append(ob_word)
#     subject_type.append(sub_type)
#     object_type.append(ob_type)
  
#   out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],
#                               'subject_type' : subject_type, 'object_type': object_type})
#   out_dataset.id = out_dataset.index
#   # print(out_dataset)
#   return out_dataset

def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  subject_type= []
  object_type= []
  subject_start= []
  object_start= []
  subject_end= []
  object_end= []
  

  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    # i = i[1:-1].split(',')[0].split(':')[1]
    sub_entity = eval(i)
    ob_entity = eval(j)
    sub_word= sub_entity['word']
    sub_type = sub_entity['type']
    # j = j[1:-1].split(',')[0].split(':')[1]
    ob_word = ob_entity['word']
    ob_type = ob_entity['type']

    ob_start = ob_entity['start_idx']
    sub_start = sub_entity['start_idx']

    ob_end = ob_entity['end_idx']
    sub_end = sub_entity['end_idx']


    subject_entity.append(sub_word)
    object_entity.append(ob_word)
    subject_type.append(sub_type)
    object_type.append(ob_type)
    subject_start.append(sub_start)
    object_start.append(ob_start)
    subject_end.append(sub_end)
    object_end.append(ob_end)
  
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],
                              'subject_type' : subject_type, 'object_type': object_type,'sub_start' : subject_start,'ob_start':object_start,'sub_end' :subject_end ,'ob_end': object_end})
  out_dataset.id = out_dataset.index
  return out_dataset


def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)
  
  return dataset



def change_word(sentence, sub_word, sub_type , ob_word, ob_type):
  sentence = re.sub(rf'{sub_word}',f'<{sub_type}>',sentence)
  sentence = re.sub(rf'{ob_word}',f'<{ob_type}>',sentence)

  return sentence




def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = ''
    temp = '<ob> ' + e01 + ' [SEP] ' + '<sub> ' + e02
    concat_entity.append(temp)
  
  
  # dataset['sentence'] = dataset['id'].apply(lambda x: change_word(dataset.sentence.loc[x],dataset.subject_entity.loc[x],
  #                                                                 dataset.subject_type.loc[x],dataset.object_entity.loc[x], dataset.object_type.loc[x]))
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=512,
      add_special_tokens=True,
      )
  return tokenized_sentences

# def tokenized_dataset(dataset, tokenizer):
#   """ tokenizer에 따라 sentence를 tokenizing 합니다."""
#   concat_entity = []
#   for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
#     temp = ''
#     temp = e01 + '[SEP]' + e02
#     concat_entity.append(temp)
#   tokenized_sentences = tokenizer(
#       concat_entity,
#       list(dataset['sentence']),
#       return_tensors="pt",
#       padding=True,
#       truncation=True,
#       max_length=256,
#       add_special_tokens=True,
#       )
#   return tokenized_sentences
