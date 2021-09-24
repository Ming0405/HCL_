import torch
from torch import nn
from torch.nn import functional as F
import sklearn, scipy
import numpy as np
from tqdm import trange, tqdm
from sklearn.cluster import KMeans
import random
from math import floor

class FSC(nn.Module):
    """FSC the main class of fuzzy semantic cells
    """
    def __init__(self, M:int) -> None:
        """
        M: the number of classes
        """
        super().__init__()
        self.M = M
    
    def setP(self, mode:int, data:torch.Tensor, labels:torch.Tensor):
        return {
            1: self.setP1,
            2: self.setP2,
            3: self.setP3,
            4: self.setP4,
        }[mode](data, labels)

    @torch.no_grad()  
    def setP1(self, data:torch.Tensor, labels:torch.Tensor) -> None:
        """setP1 the first methond to initialized prototypes. 
                    Kmeans each category.

        Args:
            data (torch.Tensor): data
            labels (torch.Tensor): labels
        """
        num_clusters = 10
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        result = []
        rearrage_labels = []
        for l in labels.unique():
            kmeans.fit(data[labels==l, :].numpy())
            result.append( torch.from_numpy(kmeans.cluster_centers_) )
            rearrage_labels += [l,] * num_clusters
            
        self.P = nn.Parameter( torch.cat(result, dim=0) )
        self.label_belonging = torch.stack( rearrage_labels, dim=0 )
        
    @torch.no_grad()  
    def setP2(self, data: torch.Tensor, labels: torch.Tensor):
        """setP2 the second methond to initialized prototypes.
                    Kmeans the whole dataset

        Args:
            data (torch.Tensor): data
            labels (torch.Tensor): labels
        """
        n = 10
        num_clusters = n * self.M
        kmeans = sklearn.cluster.KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(data.numpy())
        result = []
        rearrage_labels = []
        centers = kmeans.cluster_centers_
        for l in labels.unique():
            d = torch.cdist( torch.from_numpy(centers)[None, :, :],  data[None, labels==l, :])[0]
            idx = d.min(dim=1)[0].topk(k=n, largest=False)[1]
            result.append( torch.from_numpy(centers[idx, :]) )
            rearrage_labels += [l,]*n
            centers[idx, :] = np.Inf
        
        self.P = nn.Parameter( torch.cat(result, dim=0) )
        self.label_belonging = torch.stack( rearrage_labels, dim=0 )
    
    @torch.no_grad()  
    def setP3(self, data: torch.Tensor, labels: torch.Tensor):
        """setP3 the third methond to initialized prototypes.
                    Randomly select prototypes

        Args:
            data (torch.Tensor): data
            labels (torch.Tensor): labels
        """
        num_clusters = 10
        result = []
        rearrage_labels = []
        for l in labels.unique():
            result.append( data[labels==l, :][:num_clusters] )
            rearrage_labels += [l,] * num_clusters
        self.P = nn.Parameter( torch.cat(result, dim=0) )
        self.label_belonging = torch.stack( rearrage_labels, dim=0 )
            
    @torch.no_grad()  
    def setP4(self, data: torch.Tensor, labels: torch.Tensor):
        """setP4 the forth methond to initialized prototypes.
                    the mass center of each category

        Args:
            data (torch.Tensor): data
            labels (torch.Tensor): labels
        """
        result = []
        rearrage_labels = []
        for l in labels.unique():
            result.append( data[labels==l, :].mean(dim=0, keepdim=True) )
            rearrage_labels += [l,]
        self.P = nn.Parameter( torch.cat(result, dim=0) )
        self.label_belonging = torch.stack( rearrage_labels, dim=0 )

    @torch.no_grad()  
    def setSigma(self, data: torch.Tensor, labels: torch.Tensor):
        """setSigma set $\sigma$ as 1/3 of the average distance

        Args:
            data (torch.Tensor): data
            labels (torch.Tensor): labels
        """
        result = []
        for p, l in zip(self.P, self.label_belonging):
            result.append( torch.dist(p, data[labels==l, :]).mean()/3 )
        self.sigma = nn.Parameter( torch.stack(result, dim=0) )
        
    def forward(self, data: torch.Tensor, labels: torch.Tensor) -> torch.tensor:
        total_loss = torch.tensor(0.)
        for l in labels.unique():
            d = torch.cdist(self.P[None, :, :], data[None, labels==l, :])[0].pow(2) # |P| num_samples
            mu = torch.exp( d.div( -self.sigma[:, None].pow(2).mul(2) ) ) # |P| num_samples
            mu_i = mu.sum(dim=1) # |P| 1
            mapping = F.one_hot(self.label_belonging, num_classes=max(self.label_belonging)+1)  # |P| |C|
            result = torch.mv(mapping.transpose(0, 1).float(), mu_i) # |C|
            total_loss += F.cross_entropy(result[None, :], l[None]) # the conditional entropy
        return total_loss
    
    @torch.no_grad()
    def test(self, data:torch.Tensor) -> torch.Tensor:
        d = torch.cdist(self.P[None, :, :], data[None, :, :])[0].pow(2) # |P| num_samples
        mu = d.div(-self.sigma[:, None].pow(2).mul(2)).exp() # |P| num_samples
        idx = mu.max( dim=0 )[1]
        return self.label_belonging[idx]
    
if __name__ == "__main__":
    device = torch.device('cpu')
    train_data   = torch.from_numpy(np.loadtxt('./pendigits_sta4_train.csv', delimiter=',')).float()
    train_labels = torch.from_numpy(np.loadtxt('./pendigits_label_train.csv')).long()

    test_data   = torch.from_numpy(np.loadtxt('./pendigits_sta4_test.csv', delimiter=',')).float()
    test_labels = torch.from_numpy(np.loadtxt('./pendigits_label_test.csv')).long()

    data = torch.cat([train_data, test_data], dim=0)
    labels = torch.cat([train_labels, test_labels], dim=0)
    
    for P in [4,1,2,3]:
        for ratio in [0.1, 0.3, 0.5, 0.7,] * 10:
            idx = list(range(len(data)))
            random.shuffle(idx)
                
            num = floor(len(data) * ratio)
            # training set
            train_data = data[idx[:num], :]
            train_labels = labels[idx[:num]]
            # testing set
            test_data = data[idx[num:], :]
            test_labels = labels[idx[num:]]

            # initialize P and \sigma
            fsc = FSC(len(train_labels.unique()), train_data.size(1))
            fsc.setP(P, train_data, train_labels)
            fsc.setSigma(train_data, train_labels)

            # prepape to device
            fsc = fsc.to(device)
            train_data = train_data.to(device)
            train_label = train_labels.to(device)
            test_data = test_data.to(device)
            test_labels = test_labels.to(device)

            optimizer = torch.optim.Adam(fsc.parameters())

            # training
            loss = torch.tensor(0.)
            max_acc = 0.
            for ii in range(2000):
                # update
                loss = fsc(train_data, train_label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print loss
                loss = loss.item()
                train_acc = fsc.test(train_data).eq(train_labels).sum().div(len(train_label)).item()
                test_acc  = fsc.test(test_data.float()).cpu().eq(test_labels).sum().div(len(test_labels)).item()

                print(f"\r {ii} {loss:.4} {train_acc:.2f} {test_acc:.2f}  ", end="")
            
            print(f"P:{P}, ratio:{ratio}, train_acc:{train_acc}, test_acc:{test_acc},")
