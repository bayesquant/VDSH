import os
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from datasets import *
from utils import *
from models.VDSH_S import VDSH_S

#########################################################################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"]="3"

#########################################################################################################

train_set = Newsgroups20Dataset('dataset/ng20', subset='train', download=True, bow_format='tf')
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=64, shuffle=True)
test_set = Newsgroups20Dataset('dataset/ng20', subset='test', download=True, bow_format='tf')
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=64, shuffle=True)

#########################################################################################################
y_dim = train_set.num_classes()
num_bits = 32
num_features = train_set[0][0].size(0)

#########################################################################################################
model = VDSH_S('ng20', num_features, num_bits, y_dim, device, dropoutProb=0.1)
model.to(device)

num_epochs = 25

optimizer = optim.Adam(model.parameters(), lr=0.001)
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
