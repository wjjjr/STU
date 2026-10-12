

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tltorch import *
import math
from sparsemax import Sparsemax



class Forward_Attention_sparse(nn.Module):
    def __init__(self, patch_shape, patches, dim, dim2, attn_drop=0):
        super().__init__()
        self.dim = dim
        self.patches = patches
        self.dim2 = dim2
        self.qw = nn.Linear(dim, self.dim2, bias=False)
        self.kw = nn.Linear(dim, self.dim2, bias=False)
        self.att_drop=nn.Dropout(attn_drop)
        self.bn=nn.BatchNorm2d(self.patches)
        self.softmax=nn.Softmax(dim=2)
        self.sparsemax=Sparsemax(dim=2)
        
    def forward(self, x):
        B, N, C = x.shape
        Q=self.bn( self.qw(input=x).unsqueeze(-1) ).squeeze(-1)
        K=self.bn(self.kw(input=x).unsqueeze(-1) ).squeeze(-1)
        Q = F.normalize(Q, p=2, dim=2)
        K = F.normalize(K, p=2, dim=2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dim)
        scores = self.att_drop(scores)
        scores = self.sparsemax(scores)
        return scores


class Forward_Attention_soft(nn.Module):
    def __init__(self, patches, dim, attn_drop=0.2):
        super().__init__()
        self.dim = dim
        self.patches = patches
        self.qw = nn.Linear(dim, dim, bias=False)
        self.vw = nn.Linear(dim, dim, bias=False)
        self.att_drop=nn.Dropout(attn_drop)
        self.bn=nn.BatchNorm2d(self.patches)
        self.softmax=nn.Softmax(dim=2)

    def forward(self, x):
        B, N, C = x.shape
        assert (C == self.dim and N == self.patches)
        Q=self.bn( self.qw(input=x).unsqueeze(-1) ).squeeze(-1)
        V=self.bn(self.vw(input=x).unsqueeze(-1) ).squeeze(-1)

        scores = torch.matmul(Q, V.transpose(-2, -1)) / math.sqrt(self.dim)
        scores = self.att_drop(scores)
        scores = self.softmax(scores)
        return scores

class STU(nn.Module):
    def __init__(self, input_shape=(20, 20, 20), subtensor_size=(5,5,5), stride=(3,3,3),
                 feature_base=(5,5,5), output_rank=(20,20,20), dropout_rate=0.01, depth=1,part1_init=None,part2_init=None,  new_output_rank=4,hsi_max=0,dim2 = 0):
        super(STU, self).__init__()

        self.input_shape = input_shape
        self.subtensor_size = subtensor_size
        self.stride = stride
        self.feature_base = feature_base
        self.part1_init = part1_init
        self.part2_init = part2_init
        self.output_rank = output_rank
        self.dropout_rate = dropout_rate
        self.depth = depth
        self.hsi_max = hsi_max
        self.new_output_rank=new_output_rank
        
        self.sparsemax = Sparsemax(dim=-1)
        self.x_dim = int((input_shape[0] - subtensor_size[0]) / stride[0] + 1)
        self.y_dim = int((input_shape[1] - subtensor_size[1]) / stride[1] + 1)
        self.z_dim = int((input_shape[2] - subtensor_size[2]) / stride[2] + 1)


        self.patches = self.x_dim * self.y_dim * self.z_dim
        self.dim = feature_base[0]*feature_base[1]*feature_base[2]
        self.dim2 = self.dim
        # self.dim2 = dim2
        self.attention_blocks = nn.ModuleList([Forward_Attention_sparse(patch_shape=subtensor_size[0]*subtensor_size[1]*subtensor_size[2],patches=self.patches, dim=self.dim,dim2=self.dim2, attn_drop=dropout_rate) for _ in range(self.depth)])
        self.feature_dim1 = self.x_dim * self.x_dim
        self.feature_dim2 = self.y_dim * self.y_dim
        self.feature_dim3 = self.z_dim * self.z_dim
        self.decoder_part1 = TCL(input_shape=[self.feature_dim1, self.feature_dim2, self.feature_dim3],
                             rank=[output_rank[0], output_rank[1], new_output_rank]) 
        self.decoder_part2 = nn.Linear(new_output_rank, output_rank[2], bias=False)
        self.decoder_part2.weight = nn.Parameter(part2_init)
        self.drop = nn.Dropout(p=dropout_rate)

    def forward_features(self,x):
        for blk in self.attention_blocks:
            x = blk(x)
        return x, x

    def get_U(self):
        U_list =[]
        for i in self.decoder_part1.parameters():
            U_list.append(i.detach())
        return U_list

    @staticmethod
    def cut_tensor_into_sliding_patches(original_tensor, subtensor_size, stride):
        original_tensor = original_tensor.squeeze(0)
        oritensor_size = original_tensor.shape
        dim0_patches = (oritensor_size[0] - subtensor_size[0]) // stride[0] + 1
        dim1_patches = (oritensor_size[1] - subtensor_size[1]) // stride[1] + 1
        dim2_patches = (oritensor_size[2] - subtensor_size[2]) // stride[2] + 1

        num_patch = dim0_patches * dim1_patches * dim2_patches
        result_tensor = torch.empty((num_patch,) + subtensor_size)


        count = 0
        for i in range(0, oritensor_size[0] - subtensor_size[0] + 1, stride[0]):
            for j in range(0, oritensor_size[1] - subtensor_size[1] + 1, stride[1]):
                for k in range(0, oritensor_size[2] - subtensor_size[2] + 1, stride[2]):
                    subtensor = original_tensor[i:i + subtensor_size[0], j:j + subtensor_size[1],
                                k:k + subtensor_size[2]]
                    result_tensor[count] = subtensor
                    count += 1
        return result_tensor.unsqueeze(0)
    
    def forward(self, x_full,current_epoch=0):
        patches = x_full.cuda()
        x1 = patches
        x2 = x1.squeeze(0)
        x2 = x2.contiguous().view(self.patches, -1)
        x2 = x2.unsqueeze(0)
        x3, att_map = self.forward_features(x2)
        x3 = x3.view(self.x_dim, self.y_dim, self.z_dim,self.x_dim, self.y_dim, self.z_dim)
        x3 = x3.permute(0,3,1,4,2,5)
        x4 = x3.reshape(1,self.feature_dim1,self.feature_dim2,self.feature_dim3)
        part1 = self.decoder_part1(x4)  
        part1 = part1.squeeze(0)         
        part1 = F.softmax(part1, dim=-1)
        part = self.decoder_part2(part1)
        part2 =  self.decoder_part2.weight.T
        return (part1.squeeze(0), part2, part), att_map

def total_variation(input):
    sum_axis = None
    pixel_dif1 = input[0,1:, :, :] - input[0,:-1, :, :]
    pixel_dif2 = input[0,:, 1:, :] - input[0,:, :-1, :]
    pixel_dif3 = input[0,:, :, 1:] - input[0,:, :, :-1]
    tot_var = (
        torch.sum(torch.abs(pixel_dif1), axis=sum_axis) +
        torch.sum(torch.abs(pixel_dif2), axis=sum_axis) +
        torch.sum(torch.abs(pixel_dif3), axis=sum_axis) )
    return tot_var

def loss_fn_mse(outputs, observation_truth, add_TV_regu=False):
    pred=(outputs)
    observation_truth=observation_truth
    num_obser=outputs.numel()
    if add_TV_regu:
        alpha = 1e-8
        return torch.sum((pred-observation_truth)**2)/num_obser + alpha*total_variation(outputs)
    else:
        return torch.sum((pred-observation_truth)**2)/num_obser 





if __name__ == '__main__':
    pass

