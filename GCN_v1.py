
# coding: utf-8

# In[1]:

import pickle
from rdflib import Graph
import torch
import gzip
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, BatchNorm,  TopKPooling, global_add_pool, global_max_pool
from tqdm import tqdm


# In[2]:

#load data, feature vector for each 
f=open('dataset_1.pkl','rb')
dataset=pickle.load(f)
f.close()
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [20747, 3000])
loader=DataLoader(train_dataset,batch_size=10)
test_loader=DataLoader(test_dataset,batch_size=10)


# In[7]:

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = GCNConv(1, 30)
        self.pool1 = TopKPooling(30, ratio=0.4)
        self.conv2 = GCNConv(30, 20)
        self.pool2 = TopKPooling(20, ratio=0.2)
        self.conv3 = GCNConv(20, 10)
        self.pool3 = TopKPooling(10, ratio=0.1)

        self.lin1 = torch.nn.Linear(1380, 80)
        self.lin2 = torch.nn.Linear(80, 64)
        self.lin3 = torch.nn.Linear(64, 1)#2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)

        b=data.y.shape[0]
        x = F.relu(self.lin1(x.view(b,-1)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = torch.sigmoid(self.lin3(x))
        return x


# In[8]:

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
for parameter in model.parameters():
    print(parameter.shape)


# In[ ]:

#mostly adopted from PyTorch Geometric's example
def train(epoch):
    print('yoo')
    model.train()
    loss_all = 0
    loss_=torch.nn.BCELoss()
    for data in (loader):
        label=data.y
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_(output, data.y.view(output.shape[0],1))
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


def test(loader,thresh):
    model.eval()
    correct = 0
    print('Test!')
    for data in (loader):
        data = data.to(device)
        output = model(data)
        b,_=output.shape
        data.y=data.y.reshape(output.shape[0],1)
        correct+=sum((output>=thresh) & (data.y>0))+sum((data.y<1) & (output<thresh))
    return float(correct) / float(len(loader.dataset))


for epoch in range(1, 201):
    test_acc = test(test_loader,0.5)
    loss = train(epoch)
    train_acc = test(loader,0.5)
    test_acc = test(test_loader,0.5)
    print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
          format(epoch, loss, train_acc, test_acc))


# In[11]:

len(test_loader.dataset)


# In[ ]:



