import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from timm.models.vision_transformer import PatchEmbed, Block
from swin_transformer_v2 import SwinTransformerV2

import os, sys
import datetime
import time

import pandas as pd

import scipy.io
from scipy.fftpack import fft, dct, ifft, idct
from scipy import linalg
from scipy.interpolate import NearestNDInterpolator
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt

import numpy as np 
from numpy import linalg 

from tqdm import tqdm

from pyinstrument import Profiler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--results_path", type=str, default='./results/', 
                        help='The path of results')
    parser.add_argument("--batchsize", type=int, default=2, 
                        help='batchsize')

    # # 其他：
    parser.add_argument("--cuda", type=int, default=0, 
                        help='the number of cuda')
                
    # # Training 
    parser.add_argument('--lr_init', type=float, default=1e-4,
                        help='The initial learning rate')
    parser.add_argument('--lr_final', type=float, default=1e-5,
                        help='The final learning rate')
    parser.add_argument('--lr_delay_steps', type=float, default=0,
                        help='Tthe number of steps at the begining of training to reduce the learning rate by lr_delay_mult')    
    parser.add_argument('--lr_delay_mult', type=float, default=1,
                        help='A multiplier on the learning rate when the step is < lr_delay_steps')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='the number for cycling.')
    parser.add_argument('--evaluate_every', type=int, default=100,
                        help='the number of steps to save a checkpoint.')
    
    parser.add_argument('--print_every', type=int, default=10,
                        help='the number of steps between reports to tensorboard.')   
    
    return parser


class dataset(Dataset):
    def __init__(self, datatype, index_array, ratio, mean, std, device):
        
        Spectrum = scipy.io.loadmat('./data/spectrum_ASTER.mat')
        wavelength_Spectrum = Spectrum['w_ref']
        wavelength_re = 1e-3 * np.linspace(7000,14000,18*8)
        wavelength_filter = 1e-3 * np.array([7220,7670,7960,8330,8700,9070,9480,9810,10180,10550,10920,11290,11660,12030,12400,12770,13140])
        response_function = np.exp( - (wavelength_Spectrum.reshape(-1,1) - wavelength_filter.reshape(1,-1))**2 / ((0.3)**2) )
        SR_function = np.exp( - (wavelength_Spectrum.reshape(-1,1) - wavelength_re.reshape(1,-1))**2 / ((0.3 / 8)**2) )
        Spectrum = Spectrum['spectrum']
        intensity = response_function.T @ Spectrum
        spectrum_re = SR_function.T @ Spectrum

        self.wavelength_Spectrum = wavelength_Spectrum
        self.wavelength_re = wavelength_re
        self.wavelength_filter = wavelength_filter

        # mean_intensity = np.mean(intensity, axis = 1)
        # std_intensity = np.std(intensity - mean_intensity[:,None], axis = 1)
        # intensity = (intensity - mean_intensity[:,None]) / std_intensity[:,None]

        mean_intensity = np.mean(intensity)
        std_intensity = np.std(intensity - np.mean(intensity))
        intensity = (intensity - mean_intensity) / std_intensity

        mean_spectrum_re = np.mean(spectrum_re)
        std_spectrum_re = np.std(spectrum_re - np.mean(spectrum_re))
        spectrum_re = (spectrum_re - mean_spectrum_re) / std_spectrum_re

        mean_Spectrum = np.mean(Spectrum)
        std_Spectrum = np.std(Spectrum - np.mean(Spectrum))
        Spectrum = (Spectrum - mean_Spectrum) / std_Spectrum

        intensity = intensity.T
        spectrum_re = spectrum_re.T
        Spectrum = Spectrum.T

        # plt.plot(wavelength_re, spectrum_re[:,::100])
        # plt.plot(wavelength_filter, intensity[:,::100])
        # plt.plot(wavelength_Spectrum, Spectrum[:,::100])
        
        if (datatype == 'train') | (datatype == 'val'):

            self.mean = np.mean(intensity)
            self.std = np.std(intensity - self.mean)
            if (datatype == 'train'):
                N = intensity.shape[0]
                index = np.random.permutation(N)
                N_train = round(0.9 * N)
                N_val = N - N_train
                train_index = index[0:(N_train)]
                val_index = index[N_train:]
                self.intensity = intensity[train_index,:]
                self.spectrum_re = spectrum_re[train_index,:]
                self.Spectrum = Spectrum[train_index,:]
                self.index_array = val_index
                
                a = 1
            elif (datatype == 'val'):
                self.intensity = intensity[index_array,:]
                self.spectrum_re =  spectrum_re[index_array,:]
                self.Spectrum = Spectrum[index_array,:]


            a = 1

        elif datatype == 'test':
            
            self.intensity = intensity[:,:]
            self.spectrum_re =  spectrum_re[:,:]
            self.Spectrum = Spectrum[:,:]
            a = 1

    def __getitem__(self, index):
        Spectrum = self.Spectrum[index,:]
        spectrum_re = self.spectrum_re[index,:]
        intensity = self.intensity[index,:]

        wavelength_Spectrum = self.wavelength_Spectrum
        wavelength_re = self.wavelength_re
        wavelength_filter = self.wavelength_filter
        
        return intensity, spectrum_re, Spectrum, wavelength_filter, wavelength_re, wavelength_Spectrum

    def __len__(self):
        return self.intensity.shape[0]

class SwinT(nn.Module):
    def __init__(self, device, wavelength_filter, wavelength_re):
        super(SwinT, self).__init__()
        self.swin_layer = SwinTransformerV2(img_size=8,
                            patch_size=1,
                            in_chans=1,
                            num_classes=18*8,
                            embed_dim=96,
                            depths=[2, 2, 6, 2],
                            num_heads=[3, 6, 12, 24],
                            window_size=1,
                            mlp_ratio=4,
                            qkv_bias=True,
                            drop_rate=0.0,
                            drop_path_rate=0.1,
                            ape=False,
                            patch_norm=True,
                            use_checkpoint=False,
                            pretrained_window_sizes=[0,0,0,0])
        self.wavelength_filter = wavelength_filter 
        self.wavelength_re = wavelength_re


    def interp_input(self, x):

        B, N1 = x.shape
        N2 = 64
        n2 = int(np.sqrt(N2))
        x1 = x.unsqueeze(1)
        x2 = F.interpolate(x1, size=N2, mode='linear', align_corners=False)
        x2 = x2.reshape(B,1,n2,n2)
        return x2
    
    def interp_spectrum(self, y1):
        y1 = y1.detach().cpu().numpy()
        x1 = self.wavelength_filter
        x2 = self.wavelength_re

        
        interp_func = scipy.interpolate.interp1d(x1, y1, axis=1, fill_value='extrapolate')
        y2 = interp_func(x2)
        y2 = torch.from_numpy(y2).float().to(device)

        return y2
    
    
        
    def forward(self, x):
        
        y0 = self.interp_spectrum(x)
        x = self.interp_input(x)
        x = self.swin_layer(x)
        x = x + y0
        return x


    
class convUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(convUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        n_init = 4

        self.convnet = nn.Sequential( 
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),  
            )
        
        
        embed_dim = 272 # 192, 320, 288, 272, 17
        num_heads = 8
        mlp_ratio = 4.
        depth = 2
        norm_layer = nn.LayerNorm
        self.blocks = nn.ModuleList([
            # Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        
        self.mlpnet = nn.Sequential(
            nn.Linear(embed_dim*1, 128),
            # nn.Linear(192, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),                
            nn.Linear(128, 24),
            nn.LogSoftmax(dim=1))
        
        self.concentration_mlp = nn.Sequential(
            # nn.Linear(192*8, 128),
            nn.Linear(embed_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True), 
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),    
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),    
            nn.Linear(16, 1))


    def forward(self, x, x0):
        x01 = self.convnet(x0.unsqueeze(1))
        x01 = (x01 - torch.mean(x01)) / (torch.std(x01 - torch.mean(x01)))
        x01 = x01.flatten(1).unsqueeze(0)
        x01 = x01.expand(x.shape[0],-1,-1)

        x = x.unsqueeze(1)
        x1 = self.convnet(x)
        x1 = x1.flatten(1).unsqueeze(1)

        # f = torch.cat((x1, x01), axis = 1)
        f = x1 + torch.mean(x01, axis = 1, keepdim=True)
        # f = x1 

        for blk in self.blocks:
            f = blk(f) + f
        f = f.flatten(1)
        output = self.mlpnet(f)
        # output = self.mlpnet(x1.squeeze(1))

        output_concentration = self.concentration_mlp(x1.squeeze(1))
        
        return output, output_concentration


class CNN(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(CNN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        n_init = 4

        self.convnet = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),            
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),            
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True), 
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
            )
        
        self.mlpnet = nn.Sequential(
            nn.Linear(192, 24),
            nn.LogSoftmax(dim=1))
        
        self.concentration_mlp = nn.Sequential(
            # nn.Linear(192*8, 128),
            nn.Linear(192, 1))


    def forward(self, x, x0):

        x = x.unsqueeze(1)
        x1 = self.convnet(x)
        x1 = x1.flatten(1).unsqueeze(1)

        # output = self.mlpnet(x1)
        output = self.mlpnet(x1.squeeze(1))

        output_concentration = self.concentration_mlp(x1.squeeze(1))
        
        return output, output_concentration
    
class MLP(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(MLP, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        n_init = 4

        self.convnet = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
            )
        
        self.mlpnet = nn.Sequential(
            nn.Linear(17, 128),
            # nn.Linear(192, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 18*8))


    def forward(self, x):

        x = x.unsqueeze(1)
        x1 = self.convnet(x)
        x1 = x1.flatten(1).unsqueeze(1)

        output = self.mlpnet(x1.squeeze(1))
    
        return output 
    
@torch.no_grad()
def evaluation(dataloader, model, args, lossdic, step, mode = 'val'):
    # model.train() 
    # error 
    for batch_idx, (intensity, spectrum_re, Spectrum, wavelength_filter, wavelength_re, wavelength_Spectrum) in enumerate(dataloader):
        intensity, spectrum_re, Spectrum = pre_process(intensity, spectrum_re, Spectrum)
        output = model(intensity)
        loss = F.mse_loss(output, spectrum_re)

        if batch_idx == 0:
            loss_all = 0
            intensity_all = intensity
            output_all = output
            spectrum_re_all = spectrum_re
            Spectrum_all = Spectrum
            
        else:
            loss_all += loss.item()
            intensity_all = torch.cat((intensity_all, intensity), axis = 0)
            output_all = torch.cat((output_all, output), axis = 0)
            spectrum_re_all = torch.cat((spectrum_re_all, spectrum_re), axis = 0)
            Spectrum_all = torch.cat((Spectrum_all, Spectrum), axis = 0)

        # break
    if mode == 'val':
        lossdic['loss_val'].append(loss_all)
        mdic = {'loss_val': np.array(lossdic['loss_val'])}
        scipy.io.savemat(args.logfolder + '/loss_val.mat', mdic)
    elif mode == 'test':
        lossdic['loss_test'].append(loss_all)
        mdic = {'loss_test': np.array(lossdic['loss_test'])}
        scipy.io.savemat(args.logfolder + '/loss_test.mat', mdic)

    mdic = {'loss_train': np.array(lossdic['loss_train'])}
    scipy.io.savemat(args.logfolder + '/loss_train.mat', mdic)

    

    print('\n', step, mode, loss_all)


    if step >= 1:
        a = 1

        intensity_all = intensity_all.detach().cpu().numpy()
        output_all = output_all.detach().cpu().numpy()
        spectrum_re_all = spectrum_re_all.detach().cpu().numpy()
        Spectrum_all = Spectrum_all.detach().cpu().numpy()
        wavelength_filter = wavelength_filter.detach().cpu().numpy()
        wavelength_re = wavelength_re.detach().cpu().numpy()
        wavelength_Spectrum = wavelength_Spectrum.detach().cpu().numpy()
        

        if mode == 'test': # 保存实测数据种类预测结果
            mdic = {'intensity_all':intensity_all, 'output_all':output_all, 'spectrum_re_all':spectrum_re_all, 'Spectrum_all':Spectrum_all,
                    'wavelength_filter':wavelength_filter, 'wavelength_re':wavelength_re, 'wavelength_Spectrum':wavelength_Spectrum}
            scipy.io.savemat(args.logfolder + '/results_test.mat', mdic)
            print('save experimental results.')

        if mode == 'val': # 保存仿真数据种类预测结果
            mdic = {'intensity_all':intensity_all, 'output_all':output_all, 'spectrum_re_all':spectrum_re_all, 'Spectrum_all':Spectrum_all,
                    'wavelength_filter':wavelength_filter, 'wavelength_re':wavelength_re, 'wavelength_Spectrum':wavelength_Spectrum}
            scipy.io.savemat(args.logfolder + '/results_val.mat', mdic)
            print('save validation results.')

        ind = step % intensity_all.shape[0]
        plt.figure()
        plt.plot(wavelength_filter[0,:], intensity_all[ind,:], label='Intensity')
        plt.plot(wavelength_re[0,:], output_all[ind,:], label='Reconstructed')
        plt.plot(wavelength_re[0,:], spectrum_re_all[ind,:], label='Reference spectrum')
        plt.legend()
        # plt.plot(wavelength_Spectrum[0,:], Spectrum_all[ind,:])
        plt.savefig(args.logfolder + '/' + mode + '/' + str(step) + '_' + str(ind) + '.png')
        plt.close()

        # plt.imshow(Error_c_matrix,vmin=0.1,vmax=1)
        # k = 0
        # plt.plot(Error_c_matrix[k,:])

        # k = 0
        # idx = (gas_type_all == k)
        # intensity = intensity_all[idx,:]
        # plt.plot(intensity.T)

        # k1 = 14
        # k2 = 1
        # idx_wrong = ((prediction_all == k2) & (gas_type_all[:,None] == k1))
        # intensity_wrong = intensity_all[idx_wrong[:,0],:]
        # plt.plot(intensity_wrong.T)

        # plt.imshow(confusion)
        
        # k = 8
        # plt.plot(intensity_all[gas_type_all[:,0]==k,:].T)

        # k = 0
        # plt.plot(1000 * output_concentration_all[gas_type_all==k,0])
        # plt.plot(1000 * concentration_all[gas_type_all==k])

        # plt.plot(1000 * concentration_all[gas_type_all==k], 1000 * output_concentration_all[gas_type_all==k,0],'o')

        # plt.plot(output_concentration_all * 1000)
        # plt.plot(concentration_all * 1000)

        # plt.plot(gas_type_all, label='Groundtruth')
        # plt.plot(prediction_all, label='Prediction')
        # plt.legend()

        # 画 val 数据集的结果 sort
        # idx_array = np.argsort(concentration_all)
        # concentration_all_sorted = concentration_all[idx_array]
        # output_concentration_all_sorted = output_concentration_all[idx_array,0]
        # plt.plot(output_concentration_all_sorted)
        # plt.plot(concentration_all_sorted)
        b = 1
    
def pre_process(intensity, spectrum_re, Spectrum):
    intensity = intensity.float().to(device)
    spectrum_re = spectrum_re.float().to(device)
    Spectrum = Spectrum.float().to(device)

    return intensity, spectrum_re, Spectrum

def get_output(intensity, spectrum_re, Spectrum):
    
    return intensity, spectrum_re, Spectrum
    
if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)            # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
    torch.cuda.set_device(args.cuda)
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    args.device = device

    args.expname = sys.argv[-1][8:-4]
    args.resultsname = f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S-")}{args.expname}'
    args.logfolder = f'{args.results_path}{datetime.datetime.now().strftime("%Y%m%d-%H%M%S-")}{args.expname}'
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
    if not os.path.exists(args.logfolder):
        os.makedirs(args.logfolder)
    if not os.path.exists(args.logfolder + '/train'):
        os.makedirs(args.logfolder + '/train')
    if not os.path.exists(args.logfolder + '/val'):
        os.makedirs(args.logfolder + '/val')
    if not os.path.exists(args.logfolder + '/test'):
        os.makedirs(args.logfolder + '/test')

        
    f = os.path.join(args.logfolder, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(args.logfolder, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())


    trainset = dataset(datatype = 'train', index_array = None, ratio = 0.9, mean = None, std = None, device = device)
    valset = dataset(datatype = 'val', index_array = trainset.index_array, ratio = None, mean = trainset.mean, std = trainset.std, device = device)
    testset = dataset(datatype = 'test', index_array = None, ratio = None, mean = trainset.mean, std = trainset.std, device = device)
    

    wavelength_Spectrum = trainset.wavelength_Spectrum
    wavelength_re = trainset.wavelength_re
    wavelength_filter = trainset.wavelength_filter
    # model = convUNet(n_channels=1, n_classes=29, bilinear=False)
    # model = MLP(n_channels=1, n_classes=29, bilinear=False)
    model = SwinT(device, wavelength_filter, wavelength_re)
    print(model)
    model = model.to(device)
    
    criterion = torch.nn.MSELoss(reduction='mean')
    criterion_sigma = torch.nn.BCELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    optimizer.zero_grad()

    traindataloader = DataLoader(trainset, batch_size = args.batchsize, shuffle = True, pin_memory = True, num_workers = 0)
    valdataloader = DataLoader(valset, batch_size = args.batchsize, shuffle = True, pin_memory = True, num_workers = 0)
    testdataloader = DataLoader(testset, batch_size = args.batchsize, shuffle = False, pin_memory = True, num_workers = 0)

    lossdic = {'step':[],'loss_stage':[], 'loss_train':[], 'loss_val':[], 'loss_test':[], 'lr':[],'error_c':[],'accuracy':[],'error_c_val':[],'accuracy_val':[]}
    
    step = 0
    avg_loss = 0
    optimizer.zero_grad()
    with tqdm(initial = step, total = args.max_steps) as pbar:
        while step < args.max_steps:
            
            model.train() 
            
            for batch_idx, (intensity, spectrum_re, Spectrum, wavelength_filter, wavelength_re, wavelength_Spectrum) in enumerate(traindataloader):

                intensity, spectrum_re, Spectrum = pre_process(intensity, spectrum_re, Spectrum)

                # k_temp = 0
                # idx_temp = (gas_type == k_temp)
                # intensity_temp = intensity[idx_temp, :]
                # plt.plot(intensity_temp.T.detach().cpu().numpy())

                # intensity = (intensity + intensity * torch.randn(intensity.shape, device = intensity.device) * 0.1 
                #                     + torch.max(intensity, axis=1).values[:,None] * torch.randn(intensity.shape, device = intensity.device) * 0.1 )

                output = model(intensity)
                loss = F.mse_loss(output, spectrum_re)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                avg_loss += loss.item()

                lr = args.lr_init
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                if step % args.print_every == 0 :

                    intensity, spectrum_re, Spectrum = get_output(intensity, spectrum_re, Spectrum)
                    # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    # accuracy = pred.eq(gas_type.view_as(pred)).sum().item() / pred.shape[0]
                    # error_concentration = (output_concentration[:,0] - gas_concentration).abs().sum().item() / pred.shape[0]

                    pbar.set_description(
                        f'Iter {step:06d}:'
                        + f' all_loss = {loss.item():.4f}'
                    )

                if step % args.evaluate_every == 0:
                    lossdic['step'].append(step)
                    lossdic['lr'].append(lr)
                    lossdic['loss_train'].append(loss.item())

                    model.eval()

                    # trainset evaluation
                    evaluation(valdataloader, model, args, lossdic, step, mode = 'val')

                    ## testset evaluation
                    evaluation(testdataloader, model, args, lossdic, step, mode = 'test')
                    
                    avg_loss = 0
                model.train()
                    
                # if step % args.save_every == 0:
                #     avg_loss = 0
                #     milestone = step // args.save_every
                #     model_name = f'{model_path}/model-{milestone}.pt'
                #     if step % 50000 == 0:
                #         torch.save(model, model_name)

                #     render_utils.render(model, args, img_path, result_path, milestone)
                pbar.update(1)
                step+=1

                # profiler.stop()
                # profiler.print()
                a = 1