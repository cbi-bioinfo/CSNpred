import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd 
import os
import numpy as np
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using {device} device")


n_gcn_h1 = 512
n_gcn_h2 = 64
n_fc_h1 = 1024
n_fc_h2 = 256
n_fc_h3 = 64

dp_rate = 0.5
batch_size = 128
lr_rate = 1e-4
epoch = 200

class MyBaseDataset(Dataset):
    def __init__(self, x_data, x_gcn_data, y_data, A_hat_data):
        self.x_data = x_data
        self.x_gcn_data = x_gcn_data
        self.y_data = y_data
        self.A_hat_data = A_hat_data
    def __getitem__(self, index): 
        return self.x_data[index], self.x_gcn_data[index], self.y_data[index], self.A_hat_data[index]
    def __len__(self): 
        return self.x_data.shape[0]


class UnlabeledDataset(Dataset):
    def __init__(self, x_data, x_gcn_data, A_hat_data):
        self.x_data = x_data
        self.x_gcn_data = x_gcn_data
        self.A_hat_data = A_hat_data
    def __getitem__(self, index): 
        return self.x_data[index], self.x_gcn_data[index], self.A_hat_data[index]
    def __len__(self): 
        return self.x_data.shape[0]


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
    
    def forward(self, X, A_hat):
        return self.linear(torch.mm(A_hat, X))


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, output_dim)
    
    def forward(self, X, A_hat):
        X = nn.Dropout(dp_rate)(X)
        X = self.gcn1(X, A_hat)
        X = F.relu(X)
        X = self.gcn2(X, A_hat)
        return X


class GraphClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_classes):
        super(GraphClassifier, self).__init__()
        self.gcn = GCN(input_dim, hidden_dim, output_dim)
        self.classifier = nn.Sequential(
            nn.Linear(output_dim + num_feature, n_fc_h1),
            nn.LeakyReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(n_fc_h1, n_fc_h2),
            #nn.LeakyReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(n_fc_h2, n_fc_h3),
            #nn.LeakyReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(n_fc_h3, num_classes)
            )
    
    def forward(self, X_list, A_hat_list, X_raw):
        graph_embeddings = []
        for X, A_hat in zip(X_list, A_hat_list):
            node_embeddings = self.gcn(X, A_hat)
            graph_embedding = torch.mean(node_embeddings, dim=0)
            graph_embeddings.append(graph_embedding)
        
        graph_embeddings = torch.stack(graph_embeddings, dim=0)
        residual_embeddings = torch.cat((X_raw, graph_embeddings), dim = 1)
        logits = self.classifier(residual_embeddings)
        return logits


def normalize_adjacency(A):
    I = torch.eye(A.size(0))
    A_hat = A + I  # Add self-loop
    D_hat = torch.diag(torch.sum(A_hat, dim=1) ** -0.5)
    return torch.mm(torch.mm(D_hat, A_hat), D_hat)



train_x_filename = sys.argv[1]
train_y_filename = sys.argv[2]
test_x_filename = sys.argv[3]


gcn_data_dir = "./CSN_data"

raw_x_gcn_train = np.load(os.path.join(gcn_data_dir, "gcn_neigh_train_X.npy")) 
adj_mat_train = np.load(os.path.join(gcn_data_dir, "adj_mat_neigh_train_X.npy"))
raw_y_train = pd.read_csv(train_y_filename, index_col = False)
raw_x_train = pd.read_csv(train_x_filename, index_col = False)

try : 
    del raw_x_train['array_row']
    del raw_x_train['array_col']
except :
    print("")

raw_x_gcn_test = np.load(os.path.join(gcn_data_dir, "gcn_neigh_test_X.npy")) 
adj_mat_test = np.load(os.path.join(gcn_data_dir, "adj_mat_neigh_test_X.npy"))
raw_x_test = pd.read_csv(test_x_filename, index_col = False)

try :
    del raw_x_test['array_row']
    del raw_x_test['array_col']
except :
    print("")

num_feature = raw_x_gcn_train.shape[2]
num_train = raw_x_gcn_train.shape[0]
num_test = raw_x_gcn_test.shape[0]

X_train = torch.FloatTensor(raw_x_train.values)
X_train_gcn = torch.FloatTensor(raw_x_gcn_train)
adj_mat_train = torch.FloatTensor(adj_mat_train)

for i in range(len(adj_mat_train)) :
    adj_mat_train[i] = normalize_adjacency(adj_mat_train[i])

y_train = torch.from_numpy(raw_y_train[raw_y_train.columns[0]].values)

X_test = torch.FloatTensor(raw_x_test.values)
X_test_gcn = torch.FloatTensor(raw_x_gcn_test)
adj_mat_test = torch.FloatTensor(adj_mat_test)

for i in range(len(adj_mat_test)) :
    adj_mat_test[i] = normalize_adjacency(adj_mat_test[i])


train_dataset = MyBaseDataset(X_train, X_train_gcn, y_train, adj_mat_train)
test_dataset = UnlabeledDataset(X_test, X_test_gcn, adj_mat_test)


train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

num_celltype = len(np.unique(raw_y_train[raw_y_train.columns[0]].values))


# Initialize model
model = GraphClassifier(input_dim=num_feature, hidden_dim=n_gcn_h1, output_dim=n_gcn_h2, num_classes=num_celltype)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
criterion = nn.CrossEntropyLoss()


def train_classifier(epoch, dataloader, c_model, c_loss, c_optimizer) :
    size = len(dataloader.dataset)
    correct = 0
    for batch, (X_raw, X, y, A) in enumerate(dataloader):
        X_raw, X, y, A = X_raw.to(device), X.to(device), y.to(device), A.to(device)
        pred = c_model(X, A, X_raw)
        loss = c_loss(pred, y)
        c_optimizer.zero_grad()
        loss.backward()
        c_optimizer.step()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss = loss.item()
    correct /= size
    if epoch % 10 == 0 :
        print(f"[Epoch {epoch+1}] \tTraining loss: {loss:>5f}, Training Accuracy: {(100*correct):>0.2f}%")
    return loss, correct


def test_classifier(dataloader, c_model, c_loss):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    c_model.eval()
    pred_subtype_list = []
    with torch.no_grad():
        for batch, (X_raw, X, A) in enumerate(dataloader):
            X_raw, X, A = X_raw.to(device), X.to(device), A.to(device)
            pred = c_model(X, A, X_raw)
            pred_subtype_list.append(pred.argmax(1))
    pred_subtype_list = torch.cat(pred_subtype_list, 0)
    return pred_subtype_list

# Running classification model
for t in range(epoch):
    tmp_loss, tmp_acc = train_classifier(t, train_dataloader, model, criterion, optimizer)

max_res_pred = test_classifier(test_dataloader, model, criterion)


max_res_pred = max_res_pred.detach().cpu().numpy()
np.savetxt("prediction".csv", max_res_pred, fmt="%.0f", delimiter=",")

