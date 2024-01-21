import os
import json
from tqdm import tqdm
from time import sleep
import uuid
import sys
# import pandas as pd
from modeling import *
from data import *
import csv
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

class CedrTrainning():
    """
    [ CedrTrainning ]
    For CedrTrainning
    @ GET : Returns Result
    by sisung
    """

    def __init__(self,args):
        self.args ={
            "MODEL_TYPE":args.model_type,
            "DATA_FILE_LOC":"./dataset/",
            "WEIGHT_FILE_NAME":"model//weights.p",
            "MAX_EPOCH":210,
            "LR":0.001,
            "BERT_LR":2e-5,
            "BATCH_SIZE":16,
            "BATCHES_PER_EPOCH":32,
            "GRAD_ACC_SIZE":2
        }
        self.train_model = CedrKnrmRanker().cuda()
        super(CedrTrainning, self).__init__()
        
    def isExistDoc(self,doc_obj,doc_id):
        for doc in doc_obj:
            if doc['doc_id'] == doc_id:
                return True
        return False

    def main(self,model, dataset, train_pairs, qrels, valid_run, qrelf, model_out_dir):
        params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
        non_bert_params = {'params': [v for k, v in params if not k.startswith('bert.')]}
        bert_params = {'params': [v for k, v in params if k.startswith('bert.')], 'lr': self.args['BERT_LR']}
        optimizer = torch.optim.Adam([non_bert_params, bert_params], lr=self.args['LR'])

        epoch = 0
        top_valid_score = None
        min_loss = None
        
        with open(model_out_dir +'hist.csv', 'w', newline='') as csvfile:
            fieldnames = ['time', 'loss','acc']
            csvfile.write(','.join(fieldnames) + '\n')
            for epoch in range(self.args['MAX_EPOCH']):
                loss = self.train_iteration(model, optimizer, dataset, train_pairs, qrels)
                print('train epoch={0} loss={1}'.format(epoch,loss))
                line = '{0},{1}'.format(str(epoch),str(loss))
                csvfile.write( line + '\n')
                if min_loss is None or min_loss < loss:
                    min_loss = loss
                    model.save(os.path.join(model_out_dir, 'weights.p'))
            csvfile.close()

    def train_iteration(self,model, optimizer, dataset, train_pairs, qrels):
        total = 0
        model.train()
        total_loss = 0.
        with tqdm('training', total=self.args['BATCH_SIZE'] * self.args['BATCHES_PER_EPOCH'], ncols=80, desc='train', leave=False) as pbar:
            for record in iter_train_pairs(model, dataset, train_pairs, qrels, self.args['GRAD_ACC_SIZE']):
                scores = model(record['query_tok'],
                               record['query_mask'],
                               record['doc_tok'],
                               record['doc_mask'])
                count = len(record['query_id']) // 2
                scores = scores.reshape(count, 2)
                loss = torch.mean(1. - scores.softmax(dim=1)[:, 0])  # pariwse softmax
                loss.backward()
                total_loss += loss.item()
                total += count
                if total % self.args['BATCH_SIZE'] == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                pbar.update(count)
                if total >= self.args['BATCH_SIZE'] * self.args['BATCHES_PER_EPOCH']:
                    return total_loss

    def run_model(self,model, dataset, run, runf, desc='valid'):
        rerank_run = {}
        with torch.no_grad(), tqdm(total=sum(len(r) for r in run.values()), ncols=80, desc=desc, leave=False) as pbar:
            model.eval()
            for records in iter_valid_records(model, dataset, run, self.args['BATCH_SIZE']):
                scores = model(records['query_tok'],
                               records['query_mask'],
                               records['doc_tok'],
                               records['doc_mask'])
                for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                    rerank_run.setdefault(qid, {})[did] = score.item()
                pbar.update(len(records['query_id']))
        with open(runf, 'wt') as runfile:
            for qid in rerank_run:
                scores = list(sorted(rerank_run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
                for i, (did, score) in enumerate(scores):
                    runfile.write('{qid} 0 {did} {i + 1} {score} run\n')

    
    
    def start_train(self):
        
        documents_file_names_base = "./{0}/{1}".format(self.args['DATA_FILE_LOC'],"document_base.tsv")
        querys_file_names = "./{0}/{1}/{2}".format(self.args['DATA_FILE_LOC'],self.args['MODEL_TYPE'],"querys.tsv")
        qrels_file_names ="./{0}/{1}/{2}".format(self.args['DATA_FILE_LOC'],self.args['MODEL_TYPE'],"qrels")
        train_pairs_file_names = "./{0}/{1}/{2}".format(self.args['DATA_FILE_LOC'],self.args['MODEL_TYPE'],"s.train.pairs")
        valid_run_file_names = "./{0}/{1}/{2}".format(self.args['DATA_FILE_LOC'],self.args['MODEL_TYPE'],"valid_run")
        model_output_dir = "./{0}/{1}/{2}".format(self.args['DATA_FILE_LOC'],self.args['MODEL_TYPE'],"models")
        documents_file = open(documents_file_names_base, 'r', encoding='UTF-8')
        queries_file = open(querys_file_names, 'r', encoding='UTF-8')
        qrels_file = open(qrels_file_names, 'r', encoding='UTF-8')
        train_pairs_file = open(train_pairs_file_names, 'r', encoding='UTF-8')
        valid_run_file = open(valid_run_file_names, 'r', encoding='UTF-8')
        
        dataset = read_datafiles([queries_file,documents_file])
        qrels = read_qrels_dict(qrels_file)
        train_pairs = read_pairs_dict(train_pairs_file)
        valid_run = read_run_dict(valid_run_file)
        
        os.makedirs(model_output_dir, exist_ok=True)
        self.main(self.train_model, dataset, train_pairs, qrels, valid_run, '',model_output_dir + "/")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_type", required=True,
                        help="type of model (keybert, prank, ms-marco, yake).")
    args = parser.parse_args()    
    print("args :",args)
    cedr_train = CedrTrainning(args)
    print("cedr_train args :",cedr_train.args)   
    cedr_train.start_train()