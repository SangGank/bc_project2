from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm

from train import set_seed

file_name = 'train_num6'

def inference(model, tokenized_sent, device):
  """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
  """
  dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
  model.eval()
  output_pred = []
  output_prob = []
  for i, data in enumerate(tqdm(dataloader)):
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device)
          )
    logits = outputs[0]
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
    output_prob.append(prob)
  
  return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def num_to_label(label):
  """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
  """
  origin_label = []
  with open('./code/dict_num_to_label.pkl', 'rb') as f:
    dict_num_to_label = pickle.load(f)
  for v in label:
    origin_label.append(dict_num_to_label[v])
  
  return origin_label

def load_test_dataset(dataset_dir, tokenizer):
  """
    test dataset을 불러온 후,
    tokenizing 합니다.
  """
  test_dataset = load_data(dataset_dir)
  test_label = list(map(int,test_dataset['label'].values))
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  return test_dataset['id'], tokenized_test, test_label



def main(args):
  set_seed(42)
  """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  Tokenizer_NAME = "klue/bert-base"
  tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

  tokenizer.add_special_tokens({ "additional_special_tokens": ['<PER>', '<ORG>', '<DAT>', '<LOC>', '<POH>', '<NOH>']})


  test_dataset_dir = "./data/dataset/test/test_data.csv"
  test_dataset_total = load_data(test_dataset_dir)

  test_id_total, pred_answer_total, output_prob_total=[],[],[]
  
  for sub in ['PER','ORG']:
  # for sub in test_dataset_total.subject_type.unique():
    for ob in test_dataset_total.object_type.unique():
      
      test_dataset= test_dataset_total[(test_dataset_total.object_type == ob) & (test_dataset_total.subject_type == sub)]
      test_dataset_total = test_dataset_total.drop(test_dataset.index)
      print(test_dataset['label'].values)

      test_label = list(map(int,test_dataset['label'].values))

      test_id = test_dataset['id']
      test_dataset = tokenized_dataset(test_dataset, tokenizer)


       

  ## load my model
      MODEL_NAME = f'{args.model_dir}_{ob}_{sub}' # model dir.
      print('model_name')
      print(MODEL_NAME)
      model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
      model.parameters
      model.to(device)
      model.resize_token_embeddings(len(tokenizer))


  ## load test datset
      # test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
      Re_test_dataset = RE_Dataset(test_dataset ,test_label)

      ## predict answer
      pred_answer, output_prob = inference(model, Re_test_dataset, device) # model에서 class 추론
      pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.

      test_id_total =  test_id_total + test_id.tolist()
      pred_answer_total =  pred_answer_total + pred_answer
      output_prob_total = output_prob_total + output_prob
  
  ## make csv file with predicted answer
  #########################################################
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.

  # rx = test_dataset_total['id']
  rx = pd.DataFrame({'id':test_dataset_total['id']})
  rx['pred_label'] = 'no_relation'
  pb = [1] + [0]*29
  rx['probs'] = [pb]*rx.shape[0]
  print(rx)
  output = pd.DataFrame({'id':test_id_total,'pred_label':pred_answer_total,'probs':output_prob_total,})
  
  output = pd.concat([output, rx])
  output = output.sort_values(by=['id'])

  output.to_csv(f'./code/prediction/{file_name}.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
  #### 필수!! ##############################################
  print('---- Finish! ----')
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # model dir
  parser.add_argument('--model_dir', type=str, default=f"./best_model/{file_name}")
  args = parser.parse_args()
  print(args)
  main(args)
  
