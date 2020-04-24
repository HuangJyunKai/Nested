import numpy as np
from torch import nn ,optim
from torch.nn import functional as F
from torch.autograd import Variable
import torch
from torchvision import models
import torchvision
from unetsearch import ALLSearch,ALLSearch_Decode_V1

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    x = torch.rand(size=(1,1,256,256))
    y = torch.rand(size=(1,1,256,256))
    model = ALLSearch(3,3).to(device)
    model.load_state_dict(torch.load("./ALLSearch_best.pth",map_location='cpu'))
    para = model.arch_parameters ()
    model_f = ALLSearch_Decode_V1(1,1,para).to(device)
    
    optimizer = optim.Adam(model_f.parameters(), weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    epoch_loss = 0
    for i in range(20):
        inputs = x.to(device)
        labels = y.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        outputs = model_f(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(loss.item())
        print("epoch:",i)
    
    
    