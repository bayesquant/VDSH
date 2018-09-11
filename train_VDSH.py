import os
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from datasets import *
from utils import *
from models.VDSH import VDSH

#########################################################################################################

os.environ["CUDA_VISIBLE_DEVICES"]="3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#########################################################################################################

dataset = 'agnews'
train_set = SingleLabelTextDataset('dataset/{}'.format(dataset), subset='train', download=True, bow_format='tf')
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=64, shuffle=True)
test_set = SingleLabelTextDataset('dataset/{}'.format(dataset), subset='test', download=True, bow_format='tf')
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=64, shuffle=True)

#########################################################################################################
y_dim = train_set.num_classes()
num_bits = 32
num_features = train_set[0][0].size(0)

#########################################################################################################
model = VDSH('ng20', num_features, num_bits, dropoutProb=0.1, device=device)
model.to(device)

num_epochs = 25

optimizer = optim.Adam(model.parameters(), lr=0.001)
kl_weight = 0.
kl_step = 1 / 5000.

with open('logs/VDSH/loss.log.txt', 'w') as log_handle:
    log_handle.write('epoch,step,loss,reconstr_loss,kl_loss\n')
    
    for epoch in range(num_epochs):
        avg_loss = []
        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device)
        #     y_onehot = torch.zeros((xb.size(0), y_dim))
        #     y_onehot = y_onehot.scatter_(1, yb.unsqueeze(1), 1)

            logprob_w, mu, logvar = model(xb)
            kl_loss = VDSH.calculate_KL_loss(mu, logvar)
            reconstr_loss = VDSH.compute_reconstr_loss(logprob_w, xb)
            loss = reconstr_loss + kl_weight * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            kl_weight = min(kl_weight + kl_step, 1.)
            avg_loss.append(loss.item())
            
            log_handle.write('{},{},{:.4f},{:.4f},{:.4f}'.format(epoch, step, loss.item(), 
                                                                 reconstr_loss.item(), kl_loss.item()))
        print('{} epoch:{} loss:{:.4f}'.format(model.get_name(), epoch+1, np.mean(avg_loss)))
        
        with torch.no_grad():
            train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader)
            prec = retrieve_topk(test_b.to(device), train_b.to(device), test_y.to(device), train_y.to(device), topK=100)
            print("precision at 100: {:.4f}".format(prec.item()))
