#import the package
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import roc_auc_score


#hardcoded here, not use proportional 
def downsample(dataset, proportional = 2):
    '''
    dataset: orginal pickle final
    proportional: the number of majority class/the number of minority class
    '''
    ori_patient_feature = dataset[0] #before cleaning, original data
    ori_label = dataset[1]
    ori_label_time = dataset[2]
    
    down_sample_index = np.random.randint(0,Counter(ori_label)[0],3535)
    downsampled_features = ori_patient_feature[ori_label==0][down_sample_index]
    downsampled_label = ori_label[ori_label==0][down_sample_index]
    downsampled_label_time = ori_label_time[ori_label==0][down_sample_index]
    
    new_features = np.concatenate([downsampled_features, ori_patient_feature[ori_label==1]])
    new_label = np.concatenate([downsampled_label, ori_label[ori_label==1]])
    new_label_time = np.concatenate([downsampled_label_time, ori_label_time[ori_label==1]])
    
    return new_features,new_label,new_label_time

### Helper function
def block_tri(group_size,scale, num_units, mode):
    '''
    group_size: the size of each group
    num_units: group_size x sum(scale)
    mode: the way of connection, original, shift, fully connect
    return: tensor mask
    '''
    mtrx = np.zeros((num_units, num_units))
    if mode == 'original':
        for i in range(int(num_units/group_size)):
            mtrx[i*group_size:(i+1)*group_size, i*group_size:] = 1
    elif mode == 'shift': 
        mtrx = np.zeros((num_units, num_units))
        for i in range(int(num_units/group_size)):
            refer_li = sum([[i]*i for i in scale],[])
            length = refer_li[i]
            sequence = list(np.arange(i, num_units/group_size))
            #print(sequence)
            sequence = [int(j) for j in sequence if (j - i)%length == 0]
            ##print(sequence)
            for index in sequence:
                mtrx[index*group_size:(index+1)*group_size,i*group_size:(i+1)*group_size] = 1
    return mtrx.transpose()

#def activate_index()

def activate_index(timestep, num_units, group_size, scale,index_li,batch_size,mode,input_size = 48):
    '''
    timestep: the current timestep in a sequence
    num_units: dimension of hidden layer
    group_size: number of nodes in each group
    scale: the range of update frequency
    index_li: the index of each scale start point. dictionary. scale: position
    input_size: the feature dim for patient
    return: a matrix with 0 and 1. 1 for active rows seperately for linear layer h and i
    '''
    activation_map = np.zeros((batch_size, num_units))
    if mode == 'original':
        for i in scale:
            if timestep%i ==0:
                #print(i)
                index_temp = index_li[i]
                activation_map[:, index_temp*group_size:(index_temp + 1)*group_size] = np.ones((batch_size,group_size))
    elif mode == 'shift':
        for i in scale:
            remain = timestep%i
            if remain == 0:
                index_temp = index_li[i]
                activation_map[:,index_temp*group_size:(index_temp+1)*group_size] = np.ones((batch_size,group_size))
            else:
                index_temp = i - remain + index_li[i]
                activation_map[:, index_temp*group_size:(index_temp+1)*group_size] = np.ones((batch_size,group_size))
    return torch.from_numpy(activation_map).float()
### make it a tensor

def padding_fun(data, labels, label_time):
    '''
    This is going to pad the data in the front.
    data: a batch of patient feature. now, a list (with length as batch_size) of array of T * 48
    labels: a list of zero and one
    return target: padded_data should be batch x 48 x max_length. called in the data loader function (this is different from context window since window has fixed size and can be done before)
           length: original length of each patient records
    '''
    max_length = max(label_time)
    target_data = np.array([np.pad(i,((0,max_length - len(i)),(0,0)), 'constant', constant_values = 0) for i in data])
    target_label = np.array([np.pad([labels[i]]*label_time[i],(max_length - label_time[i],0),'constant', constant_values = -1 ) for i in range(len(labels))])
    #flip
    target_label = np.flip(target_label, axis = 1)   
    #padding -1 here, so would be ignored when calculating loss
    #print(data[i].shape)
    target = np.array(data)
    #print(target.shape)#for debug
    return target_data, target_label




#loader

class MIMICDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """
    
    def __init__(self, data_list, label_list, label_time_li):
        """
        @param data_list: list of datapoints, each element is a embedding matrix
        """
        self.data_list = data_list
        self.label_list = label_list
        self.label_time_list = label_time_li
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        return (self.data_list[key], self.label_list[key], self.label_time_list[key])

#training data in this case should be an array of arrays. for each array, it is an array of size T*48 
def MIMIC_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all 
    data have the same length
    """
    data_list = []
    label_list = []
    label_time_list = []
    for data_point in batch:#batch is a list of items(selected by index), each is imdb_train[i]#which is by above get item
        data_list.append(data_point[0])
        label_list.append(data_point[1])
        label_time_list.append(data_point[2])

    #return a batch of padded data
    new_data, label_list = padding_fun(data_list, label_list, label_time_list) #should be batch x max_len x 48    batch x max_length
    return [torch.from_numpy(new_data).float(),torch.from_numpy(np.array(label_list)).long(), torch.from_numpy(np.array(label_time_list))]

def reload_data(batch_sz,patient_feature_train, label_train, label_time_train,patient_feature_val, label_val, label_time_val):
    '''
    batch_sz: pass in the batch size
    return: data loader
    ##TODO: no test loader has been said yet
    '''        
    #print(len(training_wds))
    mimic_train = MIMICDataset(patient_feature_train, label_train, label_time_train)
    mimic_val = MIMICDataset(patient_feature_val, label_val, label_time_val)
    
    train_loader = torch.utils.data.DataLoader(dataset=mimic_train,batch_size=batch_sz,collate_fn=MIMIC_collate_func,shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=mimic_val,batch_size=batch_sz,collate_fn=MIMIC_collate_func,shuffle=True)
        
    return train_loader, validation_loader




#evaluation

def evaluation_timestamp(data_loader, model, mode = 'novel'):
    '''
    this is the timestamp level prediction check
    to be filled. use loader for easy call for summarizing training acc
    '''
    pre_ones = []
    pre_zeros = []
    label_ones = []
    label_zeros = []
    model.eval()
    
    for data, label in data_loader:
        data, label = Variable(data), Variable(label)
        #label_scores = model(data, label) #I guess should be a mtrx, batch*class_num?
    
        hidden= model.init_hidden()
        if mode == 'novel':
            hidden, output = model(data, hidden)
            output = torch.stack(output, dim=1).view(-1, 2).data.numpy()
        else:
            output = model(data, hidden).data.numpy()
        #now get a list of hidden and a list of outputs
        label = label.transpose(0,1).contiguous().view(-1).data.numpy()
        
        
        #idx
        one_idx = np.where(label == 1)[0]
        zero_idx = np.where(label == 0)[0]
        keep_idx = np.concatenate((one_idx, zero_idx))
        
        label_one = list(label[one_idx])
        label_zero = list(label[zero_idx])
        output_one = list(np.argmax(np.array(output[one_idx]),axis = 1)) #now softmax, turn into class
        output_zero = list(np.argmax(np.array(output[zero_idx]),axis = 1))
        
        
        pre_ones.extend(output_one)
        pre_zeros.extend(output_zero)
        label_ones.extend(label_one)
        label_zeros.extend(label_zero)
        #print(len(pre_ones) == len(label_ones))
        #print(len(pre_zeros) == len(label_zeros))
        
    #target = list(np.array(pre) == np.array(pre))

        #print(one_idx)
    acc0 = sum(np.array(pre_zeros) == np.array(label_zeros))/len(pre_zeros)
    acc1 = sum(np.array(pre_ones) == np.array(label_ones))/len(pre_ones)
    acc =  (sum(np.array(pre_ones) == np.array(label_ones)) + sum(np.array(pre_zeros) == np.array(label_zeros)))/(len(pre_ones) + len(pre_zeros))
    print(type(label_ones), type(label_zeros), type(pre_ones),type(pre_zeros))
    auc = roc_auc_score(label_ones+label_zeros, pre_ones + pre_zeros)
    model.train()
    return acc0, acc1,acc,auc





def evaluation(data_loader, model, mode = 'novel'):
    '''
    This is the evaluation of patient level prediction.
    '''
    label_list = []
    output_list = []
    model.eval()
    
    for i, (data, label, time_list) in enumerate(data_loader):
        data = Variable(data)
        hidden= model.init_hidden()
        #print(data.size()) #[5, 75, 48]
        #print(time_list) #which is a list
        if mode == 'novel':
            hidden, ori_output = model(data, hidden)
            ##print('the hidden size is ')
            #print(hidden[0].data.size()) #batch_size * hidden dimension
            #output is a list of prediction, everyone is batch * two
            #print('output length' + str(len(output)))
            #print('output single element size' + str(output[0].size()))
            output = [np.argmax(ori_output[time_list[i] - 1][i].data.numpy()) for i in range(data.size()[0])]
            #batch size, each one pick the element
            #each prediction for each batch member
            #print(output[0]) 
        else:
            seq, last = model(data, hidden) #should be [torch.FloatTensor of size batch * 1 * 3]
            output = np.argmax(last.data.numpy(),axis = 1)
            #print(output)
        #now get a list of hidden and a list of outputs
        #print(output)
        label = label[:,0].numpy()
        #print(label == output)
        #label = label.transpose(0,1).contiguous().view(-1).data.numpy()

        label_list.extend(label)
        output_list.extend(output)
    
    label_list = np.array(label_list)
    output_list = np.array(output_list)
    
    one_idx = np.where(label_list == 1)[0]
    zero_idx = np.where(label_list == 0)[0]
    
    label_ones = label_list[one_idx] #both are array, can select index like this
    label_zeros = label_list[zero_idx]
    pre_ones = output_list[one_idx]#now softmax, turn into class
    pre_zeros = output_list[zero_idx]
        

    acc0 = sum(pre_zeros == label_zeros)/len(pre_zeros)
    acc1 = sum(pre_ones ==label_ones)/len(pre_ones)
    acc =  (sum(pre_ones == label_ones) + sum(pre_zeros == label_zeros))/(len(pre_ones) + len(pre_zeros))
    #print(type(label_ones), type(label_zeros), type(pre_ones),type(pre_zeros))
    auc = roc_auc_score(list(label_ones) + list(label_zeros), list(pre_ones) + list(pre_zeros))
    model.train()
    return acc0, acc1,acc,auc



