import argparse
import os
import pandas

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

import utils_2
from model import Model
import warnings
warnings.filterwarnings("ignore")

SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Home device: {}'.format(device))
global top
top = np.zeros((402,2))

'''
def nce_supervised_easy(out, label, K=256):
    #supervised contrastive loss baseline
    cost = torch.exp(torch.mm(out, out.t().contiguous()))
    batch = label.shape[0]
    pos_num = 0
    Nce_loss_all = 0
    for i in range(batch):
        pos_index = torch.zeros(batch).cuda()
        neg_index = torch.zeros(batch).cuda()
        ind = torch.where(label == label[i])[0]
        pos_index[ind] = 1
        neg_index = 1 - pos_index
        pos_exp = cost[i] * pos_index
        neg_exp = cost[i] * neg_index
        p_neg = neg_index/neg_index.sum()
        K_neg = np.random.choice(p_neg.shape[0], K, p=p_neg.cpu().numpy())
        K_neg_exp = neg_exp[K_neg].mean()
        index_nce = torch.where(pos_exp != 0)
        pos_num += len(index_nce[0])
        Nce_loss = -torch.log(pos_exp[index_nce] / (pos_exp[index_nce]+K_neg_exp))
        Nce_loss_all += Nce_loss.sum()
    loss = Nce_loss_all / pos_num
    print(loss)
    return loss
    


def nce_supervised_hard(out, label, beta, K=256):
    inner_product = torch.mm(out, out.t().contiguous())
    batch = label.shape[0]
    pos_num = 0
    Nce_loss_all = 0
    for i in range(batch):
        pos_index = torch.zeros(batch).cuda()
        neg_index = torch.zeros(batch).cuda()
        ind = torch.where(label == label[i])[0]
        pos_index[ind] = 1
        neg_index = 1 - pos_index
        pos_exp = torch.exp(inner_product[i]) * pos_index
        neg_exp = torch.exp(inner_product[i]) * neg_index
        weight_exp = torch.exp(beta*inner_product[i]) * neg_index
        p_neg = weight_exp/weight_exp.sum()
        p_neg = p_neg.detach()
        K_neg = np.random.choice(p_neg.shape[0], K, p=p_neg.cpu().numpy())
        K_neg_exp = neg_exp[K_neg].mean()
        index_nce = torch.where(pos_exp != 0)
        pos_num += len(index_nce[0])
        Nce_loss = -torch.log(pos_exp[index_nce] / (pos_exp[index_nce]+K_neg_exp))
        Nce_loss_all += Nce_loss.sum()
    loss = Nce_loss_all / pos_num
    print(loss)
    return loss
'''

def nce_supervised_easy(out, label, K=256):
    #supervised contrastive loss baseline
    cost = torch.exp(torch.mm(out, out.t().contiguous()))
    batch = label.shape[0]
    pos_num = 0
    Nce_loss_all = 0
    for i in range(batch):
        pos_index = torch.zeros(batch).cuda()
        neg_index = torch.ones(batch).cuda()
        ind = torch.where(label == label[i])[0]
        pos_index[ind] = 1
        #neg_index = 1 - pos_index
        pos_exp = cost[i] * pos_index
        neg_exp = cost[i] * neg_index
        num_pos = len(torch.where(pos_exp != 0)[0])
        p_neg = neg_index/neg_index.sum()
        K_neg = np.random.choice(p_neg.shape[0], num_pos*K, p=p_neg.cpu().numpy())
        K_neg_exp = neg_exp[K_neg].reshape((num_pos, K))
        K_neg_exp = K_neg_exp.mean(1)
        index_nce = torch.where(pos_exp != 0)
        pos_num += len(index_nce[0])
        Nce_loss = -torch.log(pos_exp[index_nce]/(pos_exp[index_nce] + K_neg_exp))
        Nce_loss_all += Nce_loss.sum()
    loss = Nce_loss_all / pos_num
    return loss
    

def nce_supervised_hard(out, label, beta, K=256):
    inner_product = torch.mm(out, out.t().contiguous())
    batch = label.shape[0]
    pos_num = 0
    Nce_loss_all = 0
    for i in range(batch):
        pos_index = torch.zeros(batch).cuda()
        neg_index = torch.ones(batch).cuda()
        ind = torch.where(label == label[i])[0]
        pos_index[ind] = 1
        #neg_index = 1 - pos_index
        pos_exp = torch.exp(inner_product[i]) * pos_index
        neg_exp = torch.exp(inner_product[i]) * neg_index
        weight_exp = torch.exp(beta*inner_product[i]) * neg_index
        p_neg = weight_exp/weight_exp.sum()
        p_neg = p_neg.detach()
        num_pos = len(torch.where(pos_exp != 0)[0])
        K_neg = np.random.choice(p_neg.shape[0], num_pos*K, p=p_neg.cpu().numpy())
        K_neg_exp = neg_exp[K_neg].reshape((num_pos, K))
        K_neg_exp = K_neg_exp.mean(1)
        index_nce = torch.where(pos_exp != 0)
        pos_num += len(index_nce[0])
        Nce_loss = -torch.log(pos_exp[index_nce]/(pos_exp[index_nce] + K_neg_exp))
        Nce_loss_all += Nce_loss.sum()
    loss = Nce_loss_all / pos_num
    return loss



def train(net, data_loader, train_optimizer, temperature, estimator, beta, gradient_imp):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    loss_cal_all = 0
    for pos_1, pos_2, target in train_bar:
        pos_1 = pos_1.to(device,non_blocking=True)
        feature_1, out_1 = net(pos_1)
        if estimator == 'easy':
            loss = nce_supervised_easy(out_1, target)
        if estimator == 'hard':
            loss = nce_supervised_hard(out_1, target, beta)
        if estimator == 'else':
            loss = nce_supervised_hard(out_1, out_2, target, beta, gradient_imp)
            loss_cal = nce_supervised_easy(out_1.detach(), out_2.detach(), target)
            loss_cal_all += loss_cal.item() * batch_size

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        

        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
    
    print(total_loss / total_num)
    
    if estimator == 'else':
        loss_array[epoch][0] = float(loss_cal_all / total_num)
        loss_array[epoch][1] = float(loss_cal_all / total_num)
        np.save("cifar100_HSCL_SCL_loss_beta_"+str(beta)+".npy", loss_array)
    
    return total_loss / total_num

def test_mean_classifier(net, memory_data_loader, test_data_loader, reg, dataset_name, estimator):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            out, feature = net(data.to(device, non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).contiguous()
        # [N]
        if 'cifar' in dataset_name:
            feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device) 
        elif 'stl' in dataset_name:
            feature_labels = torch.tensor(memory_data_loader.dataset.labels, device=feature_bank.device) 
        
        #print("feature_bank", feature_bank.shape)
        feature_bank_mean = torch.zeros((int(max(feature_labels))+1, feature_bank.shape[1]), device=feature_bank.device)
        

        for i in torch.unique(feature_labels):
            index = torch.where(feature_labels == int(i))[0]
            feature_bank_mean[int(i)] = feature_bank[index].mean(0)


        feature_bank_all[epoch] = feature_bank_mean.cpu().detach().numpy()
        
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            out, feature = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank_mean.T)
            #print("sim_matrix", sim_matrix.shape)
            positive_mask = torch.zeros((sim_matrix.shape), device=feature_bank.device)
            for i in range(target.shape[0]):
                positive_mask[i][int(target[i])] = 1
            
            
            positive = positive_mask * sim_matrix
            #print(positive[0])
            positive = (positive_mask * torch.exp(positive)).sum(1)
            #print(sim_matrix)
            negative = torch.exp(sim_matrix).sum(1)
            loss = -torch.log(positive / negative)
            #print(positive / negative)
            loss = loss.mean()
            
            
            pred_labels = sim_matrix.argsort()
            #print(pred_labels[:,-1])
            #print(pred_labels[:,-5:])
            #print(target.long().unsqueeze(dim=-1))

            total_top1 += torch.sum(pred_labels[:,-1] == target.long()).float()
            #print(pred_labels[:,-1] == target)
            total_top5 += torch.sum((pred_labels[:,-5:] == target.long().unsqueeze(dim=-1)).any(dim=-1).float()).item()
            #print((pred_labels[:,-5:] == target.long().unsqueeze(dim=-1)).shape)
            #print(epoch, epochs, total_num, total_top1 / total_num * 100, total_top5 / total_num * 100)
            top[epoch][0] = float(total_top1 / total_num * 100)
            top[epoch][1] = float(total_top5 / total_num * 100)
            #print(epoch, epochs, 'total_top1', total_top1, total_num, total_top1 / total_num * 100, 'total_top5', total_top5, total_top5 / total_num * 100)
            test_bar.set_description('KNN Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))
            

    return float(total_top1 / total_num * 100), float(total_top5 / total_num * 100), float(loss)

def inner_dist(net, memory_data_loader, dataset_name, estimator):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            out, feature = net(data.to(device, non_blocking=True))
            feature_bank.append(feature)

        feature_bank = torch.cat(feature_bank, dim=0).contiguous()

        if 'cifar' in dataset_name:
            feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device) 
        elif 'stl' in dataset_name:
            feature_labels = torch.tensor(memory_data_loader.dataset.labels, device=feature_bank.device) 
        
        feature_bank_mean = torch.zeros((int(max(feature_labels))+1, feature_bank.shape[1]), device=feature_bank.device)
        
        dist_all = []
        for i in range(100):
            index = torch.where(feature_labels == int(i))[0]
            feature_bank_mean[int(i)] = feature_bank[index].mean(0)
            a_expanded = feature_bank_mean[i].unsqueeze(0)
            diff = a_expanded - feature_bank[index]  # broadcasting happens here
            distances = torch.norm(diff, dim=1)
            avg_dis = distances.mean()
            dist_all.append(float(avg_dis))

        feature_bank_all[epoch] = feature_bank_mean.cpu().detach().numpy()
        #print((feature_bank_mean[0]**2).sum())
        print("inner_class distance", np.mean(dist_all))

    return np.mean(dist_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=99, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=1, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=401, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--estimator', default='hard', type=str, help='Choose loss function')
    parser.add_argument('--dataset_name', default='cifar100', type=str, help='Choose dataset')
    parser.add_argument('--beta', default=5, type=float, help='beta')
    parser.add_argument('--gradient_imp', default=True, type=bool, help='gradient_imp')
    parser.add_argument('--N', default=1, type=float, help='M_view')
    parser.add_argument('--M', default=2, type=float, help='N_view')

    # args parse
    args = parser.parse_args()
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs, estimator,beta = args.batch_size, args.epochs, args.estimator, args.beta
    M_view, N_view = args.M, args.N
    dataset_name = args.dataset_name
    #print(args.gradient_imp)
    args.gradient_imp = False
    gradient_imp = args.gradient_imp
    
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    loss_array = np.zeros((epochs, 2))

    print(dataset_name, "estimator", estimator, beta, gradient_imp, epochs)

    results = {'train_loss': [], 'train_acc_1':[], 'test_acc_1': [], 'train_mean_classifier_loss':[], 'test_loss':[]}
    # dump args

    
    # data prepare
    train_data, memory_data, test_data = utils_2.get_dataset(dataset_name, M_view, N_view)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    if dataset_name == "tiny_imagenet":
        c = 200
        feature_dim = c - 1
    else:
        c = len(memory_data.classes)
        feature_dim = c - 1
    
    # model setup and optimizer config
    model = Model(feature_dim).to(device)
    model = nn.DataParallel(model)
    #model.load_state_dict(torch.load("Cifar100_Neural_collapse_model.pth"))

    

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    #optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=0)
    
    print(dataset_name, '# Classes: {}'.format(c))
    result = np.zeros((epochs,6))
    feature_bank_all = np.zeros((epochs, c, c-1))

    # training loop 
    if not os.path.exists('../results'):
        os.mkdir('../results')
    if not os.path.exists('../results/{}'.format(dataset_name)):
        os.mkdir('../results/{}'.format(dataset_name))
    for epoch in range(0, epochs):
        np.save("Feature_bank_"+dataset_name+"_final_acc_1_"+estimator+str(beta)+'_Seed_'+str(SEED)+str(gradient_imp), feature_bank_all)
        
        inner_dis = inner_dist(model, memory_loader, dataset_name, estimator)
        train_loss = train(model, train_loader, optimizer, temperature, estimator, beta, gradient_imp)

        if (epoch) % 20 == 0:
            train_acc_1, train_acc_5, train_mean_classifier_loss = test_mean_classifier(model, memory_loader, memory_loader, beta, dataset_name, estimator)
            
        if (epoch) % 5 == 0:
            test_acc_1, test_acc_5, train_mean_classifier_loss = test_mean_classifier(model, memory_loader, test_loader, beta, dataset_name, estimator)
        
        print(epoch, float(train_acc_1), float(inner_dis))


        result[epoch][0] = float(train_loss)
        result[epoch][1] = train_acc_1
        result[epoch][2] = train_mean_classifier_loss
        result[epoch][3] = inner_dis
        result[epoch][4] = test_acc_1
        
        np.save(dataset_name+"_final_acc_1_"+estimator+str(beta)+'_Seed_'+str(SEED)+str(gradient_imp), result)
        







