from dotmap import DotMap
import numpy as np
import scipy.io
import pickle
import os
from utils import *
from tqdm import *
from scipy import sparse
from datasets import *
from nearpy.hashes import RandomBinaryProjections
import argparse

##################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpunum", help="GPU number to train the model.")
parser.add_argument("-d", "--dataset", help="Name of the dataset.")

##################################################################################################
args = parser.parse_args()

if not args.gpunum:
    parser.error("Need to provide the GPU number.")
    
if not args.dataset:
    parser.error("Need to provide the dataset.")
        
##################################################################################################
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpunum
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#########################################################################################################
dataset, data_fmt = args.dataset.split('.')
if dataset in ['reuters', 'tmc', 'rcv1']:
    single_label = False
else:
    single_label = True

#########################################################################################################
if single_label:
    train_set = SingleLabelTextDataset('dataset/{}'.format(dataset), subset='train', bow_format=data_fmt, download=True)
    test_set = SingleLabelTextDataset('dataset/{}'.format(dataset), subset='test', bow_format=data_fmt, download=True)
else:
    train_set = MultiLabelTextDataset('dataset/{}'.format(dataset), subset='train', bow_format=data_fmt, download=True)
    test_set = MultiLabelTextDataset('dataset/{}'.format(dataset), subset='test', bow_format=data_fmt, download=True)
    
#########################################################################################################
def LSH_hash(bow):
    return [int(b) for b in lshash.hash_vector(bow.toarray().squeeze())[0]]

with torch.no_grad():
    prec_results = []
    
    for nbits in [8, 16, 32, 64, 128]:
        lshash = RandomBinaryProjections('MyHash', nbits)
        lshash.reset(train_set.num_features())

        # get hash code
        train_hash = train_set.df.bow.apply(LSH_hash)
        test_hash = test_set.df.bow.apply(LSH_hash)

        # convert hash to Tensor
        train_b = torch.Tensor(list(train_hash)).type(torch.cuda.ByteTensor)
        test_b = torch.Tensor(list(test_hash)).type(torch.cuda.ByteTensor)
        
        if single_label:
            train_y = torch.Tensor(list(train_set.df.label))
            test_y = torch.Tensor(list(test_set.df.label))
        else:
            train_y = torch.from_numpy(sparse.vstack(list(train_set.df.label)).toarray())
            test_y = torch.from_numpy(sparse.vstack(list(test_set.df.label)).toarray())
            assert(train_y.size(1) == test_y.size(1))

        assert(train_b.size(0) == train_y.size(0))
        assert(test_b.size(0) == test_y.size(0))
        assert(train_b.size(1) == test_b.size(1))

        retrieved_indices = retrieve_topk(test_b.to(device), train_b.to(device), topK=100)
        prec = compute_precision_at_k(retrieved_indices, test_y.to(device), train_y.to(device), topK=100, is_single_label=single_label)

        print("bit:{} precision at 100: {:.4f}".format(nbits, prec.item()))
        prec_results.append(prec.item())

        del train_b
        del test_b
        del train_y
        del test_y
        torch.cuda.empty_cache()
    
    result = ' & '.join(['{:.4f}'.format(p) for p in prec_results])
    print(result)