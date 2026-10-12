import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import numpy as np
import scipy
import torch
import random
import torch.optim as optim
import os
import itertools
import model.net as net
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from scipy.optimize import linear_sum_assignment
device = torch.device("cuda:0")
torch.set_num_threads(4)
 
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def hyperRMSE(S,  S_est, sor):
        N = np.size(S, 0)
        S_est = S_est[:, sor]
        rmse = S - S_est
        rmse = rmse * rmse;
        rmse = np.mean(np.sqrt(np.sum(rmse, 0) / N))
        return rmse

def hyperSAD(A, A_est):
        L = np.size(A_est, 0)
        R = np.size(A_est, 1)
        meanDistance = np.inf
        p = np.array(list(itertools.permutations(np.arange(0, R), R)))
        num = np.size(p, 0)
        sadMat = np.zeros([R, R])
        for i in range(R):
            temp = A_est[:, i]
            temp = np.tile(temp, (R, 1)).T
            sadMat[i, :] = np.arccos(
                sum(temp * A, 0) / (np.sqrt(np.sum(temp * temp, 0)+1e-4) * np.sqrt(np.sum(A * A, 0))+1e-4))
        sadMat = sadMat.T
        temp = np.zeros([1, R])
        sor = p[0, :]
        Distance = meanDistance
        for i in range(num):
            for j in range(R):
                temp[0, j] = sadMat[j, p[i, j]]
            if np.mean(temp) < meanDistance:
                sor = p[i, :]
                meanDistance = np.mean(temp, 1)
                Distance = temp.copy()
        return Distance, meanDistance, sor

def train_and_evaluate(model,  optimizer, loss_fn,  x_ob, x_true, TV=False, save_interval=500):
    x_tensor = torch.FloatTensor(x_true).cuda()
    x_tensor = x_tensor.unsqueeze(0)
    x_ob_tensor = torch.FloatTensor(x_ob).cuda()
    x_ob_tensor = x_ob_tensor.unsqueeze(0)

    rmse_min = 300
    train_loss_list=[]
    rmse_list=[]
    sparsity_list=[]
    time_tick=[]
    rmse_list=[]
    sad_list=[]
    meanDistance_min = 300

    start_time = time.time()
    X_input = model.cut_tensor_into_sliding_patches(x_ob_tensor, subtensor_size=(model.subtensor_size[0], model.subtensor_size[1], model.subtensor_size[2]), stride=model.stride)
    for epoch in tqdm(range(num_epochs)):
        output, _ = model(x_full=X_input, current_epoch=epoch)
        if isinstance(output, tuple):
            part1, part2,part  = output
            part1_true_temp = part1_true.reshape(part1.shape[0]*part1.shape[1],k) 
            part1_reshaped = part1.reshape(part1.shape[0]*part1.shape[1], k)
            part1_np = part1_reshaped.detach().cpu().numpy()
            if epoch % 10 == 0:
                part2_np = part2.detach().cpu().numpy() 
                Distance, meanDistance, sor = hyperSAD(part2_np.T ,part2_true)
                rmse_value = hyperRMSE(part1_np , part1_true_temp, sor)
            reconstructed = part
            output = reconstructed.unsqueeze(0)
        loss = loss_fn(output, x_ob_tensor, add_TV_regu=TV)
        train_loss_list.append(loss.item())


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            if hasattr(model, 'decoder_part2'):
                W = model.decoder_part2.weight
                W.data[W.data < 0] = 1e-4
                W.data[W.data > 1] = 0.95

        if True:
            output_denorm = output.detach() 
            total_rmse = rmse_value
            rmse_list.append(float(total_rmse))            
            sad_list.append(float(np.mean(meanDistance)))
            if (meanDistance < meanDistance_min or total_rmse < rmse_min or epoch % 1000 == 0) and epoch % 10 == 0:
                rmse_min = total_rmse
                meanDistance_min = meanDistance
                print("--------epoch:",epoch,"loss:",loss.item(),";min_rmse:", rmse_min,";min_sad:", meanDistance_min)
                      
    return rmse_min, meanDistance_min






if __name__=="__main__":
    ############################ data preprocessing##########################
    # ## load data
    k = 4 # endmember number 
    dataset = "WashingtonDC"  # options: "JasperRidge", "WashingtonDC", "SNR", "Urban"
    if dataset == "JasperRidge":
        data_path = '../data/JasperRidge/jasperRidge.mat'
        init_path = '../data/JasperRidge/JasperRidge4EndVCAiter1init.mat'
    elif dataset == "WashingtonDC":
        data_path = '../data/WashingtonDC/WashingtonDCData.mat'
        init_path = '../data/WashingtonDC/WashingtonDCDataVCAiter1init.mat'
        k = 5 
    elif dataset == "SNR":
        # SNR = "25dB"
        data_path = '../data/SNR/syntheticDataNewSNR25dB20170601.mat'
        init_path = '../data/SNR/syntheticDataNewSNR25dB20170601VCAiter1init.mat'
        k = 6 
    elif dataset == "Urban":
        data_path = '../data/Urban/Urban.mat'
        init_path = '../data/Urban/Urban4EndVCAiter1init.mat'
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))
    data = scipy.io.loadmat(data_path)
    data_part2_init = scipy.io.loadmat(init_path)
    x_n = np.array(data['x_n']).astype('float64')
    if x_n.ndim == 2:           # (bands, pixels) -> (rows, cols, bands)
        x_n = x_n.T.reshape((int(data['nRow'][0, 0]), int(data['nCol'][0, 0]), x_n.shape[0]), order='F')
    hsi_3d = x_n
    x_true = hsi_3d.copy() 
    hsi_max = np.max(np.abs(hsi_3d))
    x_ob_norm = hsi_3d 
    ler = 0
    if dataset == "Urban":
        x_ob_norm = x_ob_norm / hsi_max
        part2_true_temp=np.array(data['endmember']).astype('float64')
        part2_true=part2_true_temp  
        [N_x,N_y,N_z] = hsi_3d.shape
        part1_true_temp=np.array(data['abundance']).astype('float64')
        part1_true=part1_true_temp 
        print("part1_true shape before reshape:", part1_true.shape)
        part1_true_temp = part1_true.reshape((N_x,N_y,k),order='F')
        part1_true=part1_true_temp 
        part1_init =  np.array(data_part2_init['o'])
        part1_init = np.array(part1_init['S'][0,0]).astype('float64').transpose(1,0)
        part1_init = part1_init.reshape((N_x,N_y,k),order='F')
        part2_init = np.array(data_part2_init['Cn']).astype('float64')
        part1_init = torch.FloatTensor(part1_init).cuda()
        part2_init = torch.FloatTensor(part2_init).cuda()
        patch_size = (0,0,0)
        st = (0,0,0)
    else:
        # "JasperRidge", "WashingtonDC", "SNR"
        if dataset == "SNR":
            part2_true_temp=np.array(data['A']).astype('float64')
        else:
            part2_true_temp=np.array(data['endmember']).astype('float64')
        part2_true=part2_true_temp  
        
        if dataset == "SNR":
            part1_true_temp=np.array(data['abundance']).astype('float64')
        else:
            x_ob_norm = x_ob_norm / hsi_max
            part1_true_temp=np.array(data['abundance']).transpose(1,0).astype('float64')
        [N_x,N_y,N_z] = hsi_3d.shape
        part1_true=part1_true_temp 
        print("part1_true shape before reshape:", part1_true.shape)
        part1_true_temp = part1_true.reshape((N_x,N_y,k),order='F')
        part1_true=part1_true_temp 
        part1_init =  np.array(data_part2_init['o'])
        part1_init = np.array(part1_init['S'][0,0]).astype('float64').transpose(1,0)
        part1_init = part1_init.reshape((N_x,N_y,k),order='F')
        part2_init = np.array(data_part2_init['Cn']).astype('float64') 
        part1_init = torch.FloatTensor(part1_init).cuda()
        part2_init = torch.FloatTensor(part2_init).cuda()
        patch_size = (0,0,0)
        st = (0,0,0)

    #hyperparameters
    num_epochs= 5000
    learning_rate1=2e-3    
    ler = learning_rate1
    print(f"data size: {N_x},{N_y},{N_z}")
    cuda_is_available=torch.cuda.is_available()
    print("torch.cuda.is_available:",cuda_is_available)
    set_random_seed(231)
    input_shape = x_ob_norm.shape
    dim2=128
    stride=(8,8,8)
    subtensor_size=(24,24,24)
    patch_size = subtensor_size
    st = stride
    output_rank=input_shape
    feature_base = subtensor_size
    dropout_rate = 0
    new_output_rank = k
    model1 = net.STU(input_shape=input_shape,
                                 subtensor_size = subtensor_size,
                                 stride=stride,
                                 feature_base=feature_base,
                                 output_rank=output_rank,
                                 dropout_rate=dropout_rate,
                                  depth=1,
                                part1_init=part1_init,
                                part2_init=part2_init,                                 
                                 hsi_max=hsi_max,
                                 new_output_rank=new_output_rank,
                                 dim2=dim2
                                 ).cuda()

    optimizer1 = optim.AdamW(model1.parameters(), lr=learning_rate1)
    loss_fn = net.loss_fn_mse
    tap_rmse_min = train_and_evaluate(model1,  optimizer1, loss_fn, x_ob_norm , x_true, TV=False)
