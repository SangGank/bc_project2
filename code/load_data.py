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
  
class RE_Dataset2(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    # item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item = {'input_ids' : self.pair_dataset['input_ids'][idx].clone().detach,
            'attention_mask': self.pair_dataset['attention_mask'][idx].clone().detach}
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
                              'subject_type' : subject_type, 'object_type': object_type,'sub_start' : subject_start,'ob_start':object_start,'sub_end' :subject_end ,
                              'ob_end': object_end,'source':dataset['source']})
  out_dataset.id = out_dataset.index
  return out_dataset


def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)
  
  return dataset



def change_word(sentence, sub_word, sub_type , ob_word, ob_type):
  sentence = re.sub(rf'{sub_word}',f' <S. {sub_type}> ',sentence)
  sentence = re.sub(rf'{ob_word}',f' <O. {ob_type}> ',sentence)

  return sentence




def change_sentence(sentence, sub_word, sub_type , ob_word, ob_type):
  sentence = re.sub(rf'{sub_word}',f' <s> <{sub_type}> </s> ',sentence)
  sentence = re.sub(rf'{ob_word}',f' <o> <{ob_type}> </o> ',sentence)

  return sentence



def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = ''
    temp = e01 + '[SEP]' + e02
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences



def tokenized_dataset7(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  
  dataset['sentence'] = dataset['id'].apply(lambda x: change_sentence(dataset.sentence.loc[x],dataset.subject_entity.loc[x],
                                                                  dataset.subject_type.loc[x],dataset.object_entity.loc[x], dataset.object_type.loc[x]))
  

  tokenized_sentences = tokenizer(
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences


def tokenized_dataset8(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  
  dataset['sentence'] = dataset['id'].apply(lambda x: change_word(dataset.sentence.loc[x],dataset.subject_entity.loc[x],
                                                                  dataset.subject_type.loc[x],dataset.object_entity.loc[x], dataset.object_type.loc[x]))
  

  tokenized_sentences = tokenizer(
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences


def change_sentence2(sentence, sub_word, sub_type , ob_word, ob_type):
  sentence = re.sub(rf'{sub_word}',f' <s> {sub_word} <{sub_type}> </s> ',sentence)
  sentence = re.sub(rf'{ob_word}',f' <o> {ob_word} <{ob_type}> </o> ',sentence)

  return sentence

def change_sentence3(sentence, sub_word, sub_type , ob_word, ob_type):
  sentence = re.sub(rf'{sub_word}',f' <s> {sub_word} <S. {sub_type}> </s> ',sentence)
  sentence = re.sub(rf'{ob_word}',f' <o> {ob_word} <O. {ob_type}> </o> ',sentence)

  return sentence

def change_sentence4(sentence, sub_word, sub_type , ob_word, ob_type):
  sentence = re.sub(rf'{sub_word}',f' @ * <S. {sub_type}> * {sub_word} ',sentence)
  sentence = re.sub(rf'{ob_word}',f' # ^ <O. {ob_type}> ^ {ob_word} ',sentence)

  return sentence





def tokenized_dataset10(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  
  dataset['sentence'] = dataset['id'].apply(lambda x: change_sentence2(dataset.sentence.loc[x],dataset.subject_entity.loc[x],
                                                                  dataset.subject_type.loc[x],dataset.object_entity.loc[x], dataset.object_type.loc[x]))
  

  tokenized_sentences = tokenizer(
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences



def tokenized_dataset11(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02, t01, t02 in zip(dataset['subject_entity'], dataset['object_entity'],dataset['subject_type'] ,dataset['object_type']):
    temp = ''
    temp =  f'<s> {e01} <{t01}> </s>' + ' [SEP] ' + f'<o> {e02} <{t02}> </o>'
    concat_entity.append(temp)
  
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences


def tokenized_dataset12(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02, t01, t02 in zip(dataset['subject_entity'], dataset['object_entity'],dataset['subject_type'] ,dataset['object_type']):
    temp = ''
    temp =  f'<s> {e01} <{t01}> </s>' + ' [SEP] ' + f'<o> {e02} <{t02}> </o> '
    concat_entity.append(temp)
  
  dataset['sentence'] = dataset['id'].apply(lambda x: change_word(dataset.sentence.loc[x],dataset.subject_entity.loc[x],
                                                                  dataset.subject_type.loc[x],dataset.object_entity.loc[x], dataset.object_type.loc[x]))
  
  
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences

def tokenized_dataset13(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02, t01, t02 in zip(dataset['subject_entity'], dataset['object_entity'],dataset['subject_type'] ,dataset['object_type']):
    temp = ''
    temp =  f'<s> {e01} <{t01}> </s>' + ' [SEP] ' + f'<o> {e02} <{t02}> </o> '
    concat_entity.append(temp)
  
  dataset['sentence'] = dataset['id'].apply(lambda x: change_sentence2(dataset.sentence.loc[x],dataset.subject_entity.loc[x],
                                                                  dataset.subject_type.loc[x],dataset.object_entity.loc[x], dataset.object_type.loc[x]))
  
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences


def tokenized_dataset14(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02, t01, t02 in zip(dataset['subject_entity'], dataset['object_entity'],dataset['subject_type'] ,dataset['object_type']):
    temp = ''
    temp =  f'<s> {e01} <S. {t01}> </s>' + ' [SEP] ' + f'<o> {e02} <O. {t02}> </o> '

    concat_entity.append(temp)
  
  dataset['sentence'] = dataset['id'].apply(lambda x: change_sentence3(dataset.sentence.loc[x],dataset.subject_entity.loc[x],
                                                                  dataset.subject_type.loc[x],dataset.object_entity.loc[x], dataset.object_type.loc[x]))

  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences

def tokenized_dataset15(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""

  concat_entity = []
  for e01, e02, t01, t02, source in zip(dataset['subject_entity'], dataset['object_entity'],dataset['subject_type'] ,dataset['object_type'], dataset['source']):
    temp = ''
    temp =  f'<s> {e01} <S. {t01}> </s>' + ' [SEP] ' + f'<o> {e02} <O. {t02}> </o> [SEP] <{source}>'

    concat_entity.append(temp)
  
  dataset['sentence'] = dataset['id'].apply(lambda x: change_sentence3(dataset.sentence.loc[x],dataset.subject_entity.loc[x],
                                                                  dataset.subject_type.loc[x],dataset.object_entity.loc[x], dataset.object_type.loc[x]))
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences



def tokenized_dataset16(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02, t01, t02 in zip(dataset['subject_entity'], dataset['object_entity'],dataset['subject_type'] ,dataset['object_type']):
    temp = ''
    temp =  f'@ * <S. {t01}> * {e01}과 # ^ <O. {t02}> ^ {e02} 사이의 관계는 무엇인가?'

    concat_entity.append(temp)
  
  dataset['sentence'] = dataset['id'].apply(lambda x: change_sentence3(dataset.sentence.loc[x],dataset.subject_entity.loc[x],
                                                                  dataset.subject_type.loc[x],dataset.object_entity.loc[x], dataset.object_type.loc[x]))

  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences



def tokenized_dataset17(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02, t01, t02 in zip(dataset['subject_entity'], dataset['object_entity'],dataset['subject_type'] ,dataset['object_type']):
    temp = ''
    temp =  f'@ * <S. {t01}> * {e01}과 # ^ <O. {t02}> ^ {e02} 사이의 관계는 무엇인가?'

    concat_entity.append(temp)

  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences




def tokenized_dataset18(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02, t01, t02 in zip(dataset['subject_entity'], dataset['object_entity'],dataset['subject_type'] ,dataset['object_type']):
    temp = ''
    temp =  f'@ * <S. {t01}> * {e01}과 # ^ <O. {t02}> ^ {e02} 사이의 관계는 무엇인가?'

    concat_entity.append(temp)
  
  dataset['sentence'] = dataset['id'].apply(lambda x: change_sentence4(dataset.sentence.loc[x],dataset.subject_entity.loc[x],
                                                                  dataset.subject_type.loc[x],dataset.object_entity.loc[x], dataset.object_type.loc[x]))

  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences


def tokenized_dataset19(dataset, tokenizer):

  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  change_type = {'PER' : '사람', 'ORG': '조직', 'DAT':'날짜', 'LOC': '지역', 'POH' : '대체어', 'NOH' : '숫자'}
  concat_entity = []
  for e01, e02, t01, t02 in zip(dataset['subject_entity'], dataset['object_entity'],dataset['subject_type'] ,dataset['object_type']):
    temp = ''
    temp =  f'{e01}와 {e02}의 관계는 {change_type[t01]}와 {change_type[t02]}의 관계이다.'

    concat_entity.append(temp)
  
  dataset['sentence'] = dataset['id'].apply(lambda x: change_sentence4(dataset.sentence.loc[x],dataset.subject_entity.loc[x],
                                                                  dataset.subject_type.loc[x],dataset.object_entity.loc[x], dataset.object_type.loc[x]))

  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences

