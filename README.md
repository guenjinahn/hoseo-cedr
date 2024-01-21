## parameter
| Hyperparameter | Value |
|----------------|-------|
| MAX_EPOCH      | 210   |
| Learning rate  | 0.001 |
| BERT_LR        | 2e-5  |
| BATCH_SIZE     | 16    |

1) install the requirements 
pip install -r requirements.txt

2) train
python train.py --model_type keybert    (keybert, prank, ms-marco, yake)

3) test
python test.py --model_type keybert    (keybert, prank, ms-marco, yake)
