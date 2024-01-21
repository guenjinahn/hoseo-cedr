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
from ir_evaluation.effectiveness import effectiveness    

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
            "WEIGHT_FILE_NAME":"models//weights.p",
            "MAX_EPOCH":210,
            "LR":0.001,
            "BERT_LR":2e-5,
            "BATCH_SIZE":16,
            "BATCHES_PER_EPOCH":32,
            "GRAD_ACC_SIZE":2
        }
        model_file_name = '{0}/{1}/{2}'.format(self.args['DATA_FILE_LOC'],self.args['MODEL_TYPE'],self.args['WEIGHT_FILE_NAME'])
        self.test_model = CedrKnrmRanker().cuda()
        self.test_model.load(model_file_name) 
        super(CedrTrainning, self).__init__()
        
    def isExistDoc(self,doc_obj,doc_id):
        for doc in doc_obj:
            if doc['doc_id'] == doc_id:
                return True
        return False

   
    def run_model(self,model, dataset, run, runf, desc='valid'):

        BATCH_SIZE = 16
        # BATCH_SIZE = 8
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
        for qid in rerank_run:
            scores = list(sorted(rerank_run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
            return scores


    def get_test_data(self):
        docs_test= {}
        docs = {}
        run = {}        
        test_querys_objs = []
        
        test_querys_file_names = "./{0}/{1}".format(self.args['DATA_FILE_LOC'],"querys_test.tsv")
        test_documents_file_names = "./{0}/{1}".format(self.args['DATA_FILE_LOC'],"document_test.tsv")
        test_documents_base_file_names = "./{0}/{1}".format(self.args['DATA_FILE_LOC'],"document_test_base.tsv")
        querys_input_file = open(test_querys_file_names, 'r',encoding='UTF-8')
        docs_input_file = open(test_documents_file_names, 'r',encoding='UTF-8')
        docs_test_input_file = open(test_documents_base_file_names, 'r',encoding='UTF-8')
                
        while 1 :
            line = docs_test_input_file.readline()
            if not line:
                break
            array_line = line.split('\t')
            docs_test[array_line[1]]=array_line[2]

        while 1:
            line = querys_input_file.readline()
            if not line:
                break
            test_querys_objs.append(line)
            
        while 1:
            line = docs_input_file.readline()
            if not line:
                break
            array_line = line.split('\t')
            docs[str(array_line[1])] = array_line[2]
            run.setdefault('1', {})[str(array_line[1])] = 0.1
        return test_querys_objs,docs,docs_test,run
    
    def start_test(self):
        test_querys_objs,docs,docs_test,run = self.get_test_data()
        
        test_rank_file_names = "./{0}/{1}/{2}".format(self.args['DATA_FILE_LOC'],self.args['MODEL_TYPE'],"rank.txt")
        test_rank_base_file_names = "./{0}/{1}/{2}".format(self.args['DATA_FILE_LOC'],self.args['MODEL_TYPE'],"rank_base.txt")
            
        test_rank_file = open(test_rank_file_names, 'w', encoding='UTF-8')
        test_rank_base_file = open(test_rank_base_file_names, 'w', encoding='UTF-8')       
    
    
        for idx,line in enumerate(test_querys_objs) :
            queries = {}
            array_line = line.split('\t')
            queries['1'] = array_line[3]
            s_doc_id = array_line[2]
            docs[str(s_doc_id)] = docs_test[str(s_doc_id)]
            run.setdefault('1', {})[str(s_doc_id)] = 0.1
            dataset = [queries, docs]
            scores = self.run_model(self.test_model, dataset, run,'', desc='rerank')
            for  i, (did, score) in enumerate(scores):
                maked_string = str(queries) + '|' + str(s_doc_id) + '|' + str(i + 1) + '|' + str(score) +'|' + docs[str(did)]
                test_rank_base_file.write('%s' % maked_string)
                if did == s_doc_id :
                    maked_string = str(line_idx) + '|' + str(line_idx) + '|' + str(s_doc_id) + '|' + str(i+1)
                    test_rank_file.write('%s\n' % maked_string)
            line_idx += 1  
        test_rank_file.close()
        test_rank_base_file.close()
        return test_rank_file_names
    
    def mrr_score(self,test_rank_file_names):
        ir = effectiveness()
        interactions = {}
        dataset2 = open(test_rank_file_names, 'r', encoding='UTF-8')
        for row in dataset2:
            row = row.strip().split('|')

            if interactions.get(row[1]) is None:
                interactions[row[1]] = {}

            if interactions[row[1]].get('visited_documents_orders') is None:
                interactions[row[1]]['visited_documents_orders'] = []
            interactions[row[1]]['visited_documents_orders'].append(row[3])
        mrr_score = ir.mrr(interactions)
        return mrr_score      

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_type", required=True,
                        help="type of model (keybert, prank, ms-marco, yake).")
    args = parser.parse_args()    
    print("args :",args)
    cedr_train = CedrTrainning(args)
    print("cedr_train args :",cedr_train.args)   
    test_rank_file_names = cedr_train.start_test()
    mrr_score = cedr_train.mrr_score(test_rank_file_names)
    print("Mean Reciprocal Rank: ",mrr_score)