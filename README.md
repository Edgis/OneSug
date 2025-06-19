OneSug: The Unified End-to-End Generative Framework for E-commerce Query Suggestion
====================================


![alt text](model.png)
<p align="center">Overall structure of the OneSug.</p>

OneSug: The Unified End-to-End Generative Framework for E-commerce Query Suggestion


This is a PyTorch implementation for [OneSug: The Unified End-to-End Generative Framework for E-commerce Query Suggestion].



Data
----------------------
We have given 10 anonymized training data samples in `data.txt`. Each sample contains three columns: prefix, searched_query, rqvae_query, chosen_sample, reject_sample, pos_score, neg_score.

Train
----------------------
Because OneSug has been applied to Kuaishou online, the code is company confidential. Therefore, we make the training loss and training pseudocode public in `main.py` to help researchers better understand and reproduce our paper. 
