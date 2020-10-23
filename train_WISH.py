import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from dataloading import *
from utils import *
from models.WISH import WISH
os.environ['CUDA_VISIBLE_DEVICES']='1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(args):
    # load data
    dataset = args.dataset
    data_fmt = args.data_fmt
    single_label_flag = args.single_label_flag
    if single_label_flag == False:
        train_set = MultiLabelTextDataset('dataset/{}'.format(dataset), subset='train', bow_format=data_fmt, download=True)
        test_set = MultiLabelTextDataset('dataset/{}'.format(dataset), subset='test', bow_format=data_fmt, download=True)
        val_set = MultiLabelTextDataset('dataset/{}'.format(dataset), subset='validation', bow_format=data_fmt, download=True)
    else:
        train_set = SingleLabelTextDataset('dataset/{}'.format(dataset), subset='train', bow_format=data_fmt, download=True)
        test_set = SingleLabelTextDataset('dataset/{}'.format(dataset), subset='test', bow_format=data_fmt, download=True)
        val_set = SingleLabelTextDataset('dataset/{}'.format(dataset), subset='validation', bow_format=data_fmt, download=True)
        
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.test_batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=args.val_batch_size, shuffle=False)

    # build model
    y_dim = train_set.num_classes()
    num_bits = args.num_bits
    topicDim = args.topicDim
    topicNum = args.topicNum
    num_features = train_set[0][0].size(0)
    print(y_dim, num_features)
    
    model = WISH(dataset, num_features, num_bits, topicDim, topicNum, device=device, dropoutProb=0.2)
    model.to(device)
    
    # train
    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr)
    kl_weight = args.kl_weight
    kl_step = args.kl_step
    alpha = args.alpha
    num_epochs = args.num_epochs

    best_precision = 0
    test_precision = 0
    best_precision_epoch = 0

    I = torch.eye(topicNum).cuda()
    
    for epoch in range(num_epochs):
        avg_loss = []
        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device)
            yb = yb.to(device)

            logprob_w, mu, topicS = model(xb, True, integration=args.integration)
            
            kl_loss = WISH.calculate_KL_loss(mu)
            reconstr_loss = WISH.compute_reconstr_loss(logprob_w, xb)
          
            loss = reconstr_loss + kl_weight * kl_loss
            loss += torch.pow(torch.norm(topicS - I), 2) * alpha
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            kl_weight = min(kl_weight + kl_step, 1.)
            avg_loss.append(loss.item())
            
        with torch.no_grad():
            # validation
            train_b, val_b, train_y, val_y = model.get_binary_code(train_loader, val_loader, False)
            retrieved_indices = retrieve_topk(val_b.to(device), train_b.to(device), topK=100)
            prec_val = compute_precision_at_k(retrieved_indices, val_y.to(device), train_y.to(device), topK=100, is_single_label=single_label_flag)
            
            # test
            train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader, False)
            retrieved_indices = retrieve_topk(test_b.to(device), train_b.to(device), topK=100)
            prec_test = compute_precision_at_k(retrieved_indices, test_y.to(device), train_y.to(device), topK=100, is_single_label=single_label_flag)
            
            
            if prec_val.item() > best_precision:
                best_precision = prec_val.item()
                best_precision_epoch = epoch
                test_precision = prec_test.item()
            
            
        print("current precision at 100, val: {:.4f}, test: {:.4f}".format(prec_val.item(), prec_test.item()))
        
        print('{} epoch:{} loss:{:.4f} Best Precision:({}){:.4f}'.format(model.get_name(), epoch, np.mean(avg_loss), best_precision_epoch, best_precision))
        
        with open("results-WISH.txt", "a") as text_file:
            print("current precision at 100, val: {:.4f}, test: {:.4f}\n".format(prec_val.item(), prec_test.item()), file=text_file)
            print('{} epoch:{} loss:{:.4f} Best Val Precision:({}){:.4f}\n'.format(model.get_name(), epoch, np.mean(avg_loss), best_precision_epoch, best_precision), file=text_file)
        
        
    with open("results-WISH.txt", "a") as text_file:
        text_file.write('======================================================\n')
        print('{} Dim:{} {} bits:{} Best Precision:({}){:.4f}\n\n'.format(dataset, topicDim, model.get_name(), num_bits, best_precision_epoch, test_precision), file=text_file)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--dataset", default='reuters', type=str)
    parser.add_argument("--data_fmt", default='tf', type=str)
    parser.add_argument("--single_label_flag", action='store_true')
    parser.add_argument("--integration", default='sum', type=str,
                        help="sum or average")

    parser.add_argument("--train_batch_size", default=100, type=int)
    parser.add_argument("--test_batch_size", default=100, type=int)
    parser.add_argument("--val_batch_size", default=100, type=int)
    
    parser.add_argument("--num_bits", default=4, type=int)
    parser.add_argument("--topicNum", default=4, type=int,
                        help="always the same as num_bits")
    parser.add_argument("--topicDim", default=50, type=int)
    
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--num_epochs", default=1000, type=int) #########
    parser.add_argument("--kl_weight", default=0., type=float)
    parser.add_argument("--kl_step", default=5e-6, type=float)
    parser.add_argument("--alpha", default=1., type=float)
    
    args = parser.parse_args()
    
    print('pytorch version: ', torch.__version__)
    print(args)
    main(args)
