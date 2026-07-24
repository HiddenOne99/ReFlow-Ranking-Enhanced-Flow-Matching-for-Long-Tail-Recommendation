# ReFlow-Ranking-Enhanced-Flow-Matching-for-Long-Tail-Recommendation

This repository contains the implementation for "ReFlow: Ranking-Enhanced Flow Matching for Long-Tail Recommendation" paper.

To reproduce the experiments, please follow the steps below:

1-Install the necessary packages using the following command:
```
conda env create --file requirements.yaml
```
<br><br><br>
2-For convenience and better reproducibility, all datasets are included in the data.zip file. Unzip them in the same directory as reflow.py script.
<br><br><br><br>
3-For each dataset, use the following commands to run the code:

ML-1M:
```
python reflow.py --train_path ml1mtrain.inter --valid_path ml1mvalid.inter --test_path ml1mtest.inter --n_steps 50 --mlp_dropout 0.4 --rank_lambda 0.0005 --rank_num_neg 1000 --batch_size 4096 --max_epochs 200 --patience 10 --lr 1e-3 --gpu 0
```
Gowalla:
```
python reflow.py --train_path gowallatrain.inter --valid_path gowallavalid.inter --test_path gowallatest.inter --n_steps 50 --mlp_dropout 0.0 --rank_lambda 0.01 --rank_num_neg 1000 --batch_size 4096 --max_epochs 200 --patience 10 --lr 1e-3 --gpu 0
```
Douban-Book:
```
python reflow.py --train_path doubantrain.inter --valid_path doubanvalid.inter --test_path doubantest.inter --n_steps 50 --mlp_dropout 0.2 --rank_lambda 0.005 --rank_num_neg 1000 --batch_size 4096 --max_epochs 200 --patience 10 --lr 1e-3 --gpu 0
```
Yelp:
```
python reflow.py --train_path yelptrain.inter --valid_path yelpvalid.inter --test_path yelptest.inter --n_steps 50 --mlp_dropout 0.3 --rank_lambda 0.01 --rank_num_neg 3000 --batch_size 4096 --max_epochs 200 --patience 10 --lr 1e-3 --gpu 0
```
Amazon-Book:
```
python reflow.py --train_path amazontrain.inter --valid_path amazonvalid.inter --test_path amazontest.inter --n_steps 50 --mlp_dropout 0.0 --rank_lambda 0.01 --rank_num_neg 3000 --batch_size 2048 --max_epochs 200 --patience 10 --lr 1e-3 --gpu 0
```
