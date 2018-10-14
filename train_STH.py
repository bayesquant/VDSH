import os
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix, csc_matrix
from tqdm import tqdm
import scipy
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import LinearSVC
from utils import *
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

max_nodes = 20
gpunum = args.gpunum
dataset, fmt = args.dataset.split('.')
data_dir = 'dataset/{}'.format(dataset)

fn = os.path.join(data_dir, 'train.NN.pkl')
df_train = pd.read_pickle(fn)
#df_train.set_index('doc_id', inplace=True)

docid2index = {docid: index for index, docid in enumerate(list(df_train.index))}

# Test data
fn = os.path.join(data_dir, 'test.NN.pkl')
df_test = pd.read_pickle(fn)
#df_test.set_index('doc_id', inplace=True)

num_train = len(df_train)
num_test = len(df_test)

r = []
c = []
d = []
row_index = 0
for idx, row in df_train.iterrows():
    r += [row_index] * max_nodes
    c += [docid2index[docid] for docid in row.top_nn[:max_nodes]]
    d += row.scores[:max_nodes]
    row_index += 1
    
weight_mat = csc_matrix((d, (r, c)), shape=(num_train, num_train))
train_bow = sparse.vstack(list(df_train.bow))
test_bow = sparse.vstack(list(df_test.bow))

class MedianHashing(object):
    def __init__(self):
        self.threshold = None
        self.latent_dim = None
    
    def fit(self, X):
        self.threshold = np.median(X, axis=0)
        self.latent_dim = X.shape[1]
        
    def transform(self, X):
        assert(X.shape[1] == self.latent_dim)
        binary_code = np.zeros(X.shape)
        for i in range(self.latent_dim):
            binary_code[np.nonzero(X[:,i] < self.threshold[i]),i] = 0
            binary_code[np.nonzero(X[:,i] >= self.threshold[i]),i] = 1
        return binary_code.astype(int)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
class STH:
    def __init__(self, num_bits, topK):
        super(STH, self).__init__()
        self.num_bits = num_bits
        self.clfs = [LinearSVC() for n in range(num_bits)]
        self.topK = topK
    
    def fit_transform(self, bow_mat, weight_mat, num_train):
        W = weight_mat
        D = np.asarray(W.sum(axis=1)).squeeze() + 0.0001 # adding damping value for a numerical stabability
        D = scipy.sparse.diags(D)
        L = D - W
    
        L = scipy.sparse.csc_matrix(L)
        D = scipy.sparse.csc_matrix(D)

        num_attempts = 0
        max_attempts = 3
        success = False
        
        while not success:
            E, Y = eigsh(L, k=self.num_bits+1, M=D, which='SM')
            success = np.all(np.isreal(Y))
            
            if not success:
                print("Warning: Some eigenvalues are not real values. Retry to solve Eigen-decomposition.")
                num_attempts += 1
            
            if num_attempts > max_attempts:
                assert(np.all(np.isreal(Y))) # if this fails, re-run fit again
                assert(False) # Check your data 
        
        Y = np.real(Y)
        Y = Y[:, 1:]
        
        medHash = MedianHashing()
        cbTrain = medHash.fit_transform(Y)    
        for b in range(0, cbTrain.shape[1]):
            self.clfs[b].fit(bow_mat, cbTrain[:, b])
        return cbTrain
    
    def transform(self, bow_mat, num_test):
        cbTest = np.zeros((num_test, self.num_bits), dtype=np.int64)
        for b in range(0, self.num_bits):
            cbTest[:,b] = self.clfs[b].predict(bow_mat)
        return cbTest

os.environ["CUDA_VISIBLE_DEVICES"]=gpunum
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if single_label:
    train_y = torch.Tensor(list(df_train.label))
    test_y = torch.Tensor(list(df_test.label))
else:
    train_y = torch.from_numpy(sparse.vstack(list(df_train.label)).toarray())
    test_y = torch.from_numpy(sparse.vstack(list(df_test.label)).toarray())
    assert(train_y.size(1) == test_y.size(1))
    
with torch.no_grad():
    prec_results = []
    
    for num_bits in [8, 16, 32, 64, 128]:
        model = STH(num_bits, None)
        train_b = model.fit_transform(train_bow, weight_mat, None)
        test_b = model.transform(test_bow, test_bow.shape[0])

        # convert hash to Tensor
        train_b = torch.Tensor(list(train_b)).type(torch.ByteTensor)
        test_b = torch.Tensor(list(test_b)).type(torch.ByteTensor)

        assert(train_b.size(0) == train_y.size(0))
        assert(test_b.size(0) == test_y.size(0))
        assert(train_b.size(1) == test_b.size(1))

        retrieved_indices = retrieve_topk(test_b.to(device), train_b.to(device), topK=100)
        prec = compute_precision_at_k(retrieved_indices, test_y.to(device), train_y.to(device), topK=100, is_single_label=single_label)

        print("bit:{} precision at 100: {:.4f}".format(num_bits, prec.item()))
        prec_results.append(prec.item())

        del train_b
        del test_b
        torch.cuda.empty_cache()
        
    result = ' & '.join(['{:.4f}'.format(p) for p in prec_results])
    print(result)