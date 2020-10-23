import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class LBSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class WISH(nn.Module):
    
    def __init__(self, dataset, vocabSize, latentDim, topicDim, topicNum, device, dropoutProb=0.):
        super(WISH, self).__init__()
        
        self.dataset = dataset
        self.hidden_dim = 1000
        self.vocabSize = vocabSize
        self.latentDim = latentDim
        self.topicDim = topicDim
        self.topicNum = topicNum
        self.dropoutProb = dropoutProb
        self.device = device
        
        self.topicBook = nn.Parameter(torch.rand(self.topicNum, self.topicDim, requires_grad=True).to(self.device))
        
        self.encoder = nn.Sequential(nn.Linear(self.vocabSize, self.hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=dropoutProb))
        
        self.h_to_mu = nn.Sequential(nn.Linear(self.hidden_dim, self.latentDim),
                                     nn.Sigmoid())
                    
        self.decoder = nn.Sequential(nn.Linear(self.topicDim, self.vocabSize),
                                     nn.LogSoftmax(dim=1))
        
    def encode(self, doc_mat, isStochastic):
        h = self.encoder(doc_mat)
        mu = self.h_to_mu(h)
        z = self.binarization(mu, isStochastic)
        return z, mu
    
    def binarization(self, mu, isStochastic):
        lb_sign = LBSign.apply
        if isStochastic:
            thresh = torch.FloatTensor(mu.size()).uniform_().to(self.device)
            return (lb_sign(mu - thresh) + 1) / 2
        else:
            return (lb_sign(mu - 0.5) + 1) / 2
        
    def forward(self, document_mat, isStochastic, integration='sum'):
        z, mu = self.encode(document_mat, isStochastic)
        if integration == 'sum':
            z_nor = z
        else:
            cnt = torch.sum(z, dim=-1) # row sum
            cnt[cnt==0] = 1.
            z_nor = z.div(cnt.view(-1, 1))
        
        topic_com = torch.mm(z_nor, self.topicBook)
        prob_w = self.decoder(topic_com)
        return prob_w, mu, torch.mm(self.topicBook, self.topicBook.t())
    
    def get_name(self):
        return "WISH"
    
    @staticmethod
    def calculate_KL_loss(mu):
        thresh = 1e-20 * torch.ones(mu.size()).cuda()
        KLD_element = mu * torch.log(torch.max(mu * 2, thresh)) + (1 - mu) * torch.log(torch.max((1 - mu) * 2, thresh))
        KLD = torch.sum(KLD_element, dim=1)
        KLD = torch.mean(KLD)
        return KLD

    @staticmethod
    def compute_reconstr_loss(logprob_word, doc_mat):
        return -torch.mean(torch.sum(logprob_word * doc_mat, dim=1))
    
    def get_binary_code(self, train, test, isStochastic):
        train_zy = [(self.encode(xb.to(self.device), isStochastic)[0], yb) for xb, yb in train]
        train_z, train_y = zip(*train_zy)
        train_z = torch.cat(train_z, dim=0)
        train_y = torch.cat(train_y, dim=0)

        test_zy = [(self.encode(xb.to(self.device), isStochastic)[0], yb) for xb, yb in test]
        test_z, test_y = zip(*test_zy)
        test_z = torch.cat(test_z, dim=0)
        test_y = torch.cat(test_y, dim=0)
        
        train_b = train_z.type(torch.cuda.ByteTensor)
        test_b = test_z.type(torch.cuda.ByteTensor)

        del train_z
        del test_z

        return train_b, test_b, train_y, test_y
