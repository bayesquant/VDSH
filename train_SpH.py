from dotmap import DotMap
import numpy as np
import scipy.io
import pickle
import os
from utils import *
from tqdm import *
from scipy import sparse
from datasets import *
import argparse
from models.SpH import *

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

train = sparse.vstack(list(train_set.df.bow))
test = sparse.vstack(list(test_set.df.bow))

if single_label:
    train_y = torch.Tensor(list(train_set.df.label))
    test_y = torch.Tensor(list(test_set.df.label))
else:
    train_y = torch.from_numpy(sparse.vstack(list(train_set.df.label)).toarray())
    test_y = torch.from_numpy(sparse.vstack(list(test_set.df.label)).toarray())
    assert(train_y.size(1) == test_y.size(1))
    
#########################################################################################################

with torch.no_grad():
    prec_results = []
    
    for nbits in [8, 16, 32, 64, 128]:
        print('Train SpH with number of bits: {} ...'.format(nbits))
        SpH_model = SpH(nbits)

        # get hash code
        train_hash = SpH_model.fit_transform(train)
        test_hash = SpH_model.transform(test)

        # convert hash to Tensor
        train_b = torch.Tensor(list(train_hash)).type(torch.ByteTensor)
        test_b = torch.Tensor(list(test_hash)).type(torch.ByteTensor)
        
        assert(train_b.size(0) == train_y.size(0))
        assert(test_b.size(0) == test_y.size(0))
        assert(train_b.size(1) == test_b.size(1))

        retrieved_indices = retrieve_topk(test_b.to(device), train_b.to(device), topK=100)
        prec = compute_precision_at_k(retrieved_indices, test_y.to(device), train_y.to(device), topK=100, is_single_label=single_label)

        print("bit:{} precision at 100: {:.4f}".format(nbits, prec.item()))
        prec_results.append(prec.item())

        del train_b
        del test_b
        torch.cuda.empty_cache()
    
    result = ' & '.join(['{:.4f}'.format(p) for p in prec_results])
    print(result)