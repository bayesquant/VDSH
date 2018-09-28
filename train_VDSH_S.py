import os
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from datasets import *
from utils import *
from models.VDSH_S import VDSH_S
import argparse

##################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpunum", help="GPU number to train the model.")
parser.add_argument("-d", "--dataset", help="Name of the dataset.")
parser.add_argument("-b", "--nbits", help="Number of bits of the embedded vector.", type=int)
parser.add_argument("--dropout", help="Dropout probability (0 means no dropout)", default=0.1, type=float)
parser.add_argument("--train_batch_size", default=100, type=int)
parser.add_argument("--test_batch_size", default=100, type=int)
parser.add_argument("--transform_batch_size", default=100, type=int)
parser.add_argument("--num_epochs", default=30, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument('--single-label', dest='pred_type', action='store_true', help='Use softmax on the output of the prediction layer.')
parser.add_argument('--multi-label', dest='pred_type', action='store_false', help='Use sigmoid on the output of the prediction layer.')
parser.set_defaults(pred_type=True)

args = parser.parse_args()

if not args.gpunum:
    parser.error("Need to provide the GPU number.")
    
if not args.dataset:
    parser.error("Need to provide the dataset.")

if not args.nbits:
    parser.error("Need to provide the dataset.")
        
##################################################################################################

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpunum
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#########################################################################################################

dataset, data_fmt = args.dataset.split('.')
train_set = SingleLabelTextDataset('dataset/{}'.format(dataset), subset='train', bow_format=data_fmt, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.train_batch_size, shuffle=True)
test_set = SingleLabelTextDataset('dataset/{}'.format(dataset), subset='test', bow_format=data_fmt, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.test_batch_size, shuffle=True)

#########################################################################################################
y_dim = train_set.num_classes()
num_bits = args.nbits
num_features = train_set[0][0].size(0)

print("Train VDSH-S model ...")
print("dataset: {}".format(args.dataset))
print("numbits: {}".format(args.nbits))
print("gpu id:  {}".format(args.gpunum))
print("dropout probability: {}".format(args.dropout))
if args.pred_type:
    print("single-label prediction.")
else:
    print("multi-label prediction.")
print("num epochs: {}".format(args.num_epochs))
print("learning rate: {}".format(args.lr))

#########################################################################################################
model = VDSH_S(dataset, num_features, num_bits, y_dim, device, dropoutProb=args.dropout, use_softmax=args.pred_type)
model.to(device)

num_epochs = args.num_epochs

optimizer = optim.Adam(model.parameters(), lr=args.lr)
kl_weight = 0.
kl_step = 1 / 5000.
pred_weight = 0.
pred_weight_step = 1 / 1000.

with open('logs/VDSH_S/loss.log.txt', 'w') as log_handle:
    log_handle.write('epoch,step,loss,reconstr_loss,kl_loss\n')
    
    for epoch in range(num_epochs):
        avg_loss = []
        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device)
            yb = yb.to(device)
        #     y_onehot = torch.zeros((xb.size(0), y_dim))
        #     y_onehot = y_onehot.scatter_(1, yb.unsqueeze(1), 1)

            logprob_w, score_c, mu, logvar = model(xb)
            kl_loss = VDSH_S.calculate_KL_loss(mu, logvar)
            reconstr_loss = VDSH_S.compute_reconstr_loss(logprob_w, xb)
            pred_loss = model.compute_prediction_loss(score_c, yb)
            
            loss = reconstr_loss + kl_weight * kl_loss + pred_weight * pred_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            kl_weight = min(kl_weight + kl_step, 1.)
            pred_weight = min(pred_weight + pred_weight_step, 150.)
            avg_loss.append(loss.item())
            
            log_handle.write('{},{},{:.4f},{:.4f},{:.4f}'.format(epoch, step, loss.item(), 
                                                                 reconstr_loss.item(), kl_loss.item()))
        print('{} epoch:{} loss:{:.4f}'.format(model.get_name(), epoch+1, np.mean(avg_loss)))
        
        with torch.no_grad():
            train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader)
            prec = retrieve_topk(test_b.to(device), train_b.to(device), test_y.to(device), train_y.to(device), topK=100)
            print("precision at 100: {:.4f}".format(prec.item()))
