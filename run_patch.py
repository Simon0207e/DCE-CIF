import h5py
import numpy as np
#import ipdb
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import io
from DCE_Vit_small import VisionTransformer as VT_65
from DCE_Vit_small_nt_113 import VisionTransformer as VT_113
import matplotlib.pyplot as plt
#import ipdb
device = torch.device("cpu")

def gen_sliding_patches(patch_size, Eh,nbase, mask=None):
    if mask != None:
        cord = (mask[0,0,:,:] == 1).nonzero(as_tuple=True)
        hw = patch_size // 2
        UL_row = torch.min(cord[0]) - hw
        if UL_row == 0:
            UL_row = 1

        UL_col = torch.min(cord[1]) - hw
        LR_row = torch.max(cord[0]) + hw
        LR_col = torch.max(cord[1]) + hw
        mask_t = torch.repeat_interleave(mask, Eh.shape[0], dim=0)
    else:
        cord = (torch.arange(Eh.shape[0]), torch.arange(Eh.shape[1]))
        hw = patch_size // 2
        UL_row = torch.min(cord[0])
        if UL_row == 0:
            UL_row = 1
        UL_col = torch.min(cord[1])
        LR_row = torch.max(cord[0])
        LR_col = torch.max(cord[1])

    # print(cord[0].shape)
    cnt_row = LR_row - UL_row - 1
    cnt_col = LR_col - UL_col - 1

    out_patches = torch.Tensor(Eh.shape[1], cnt_row * cnt_col, patch_size, patch_size)
    cnt = 0
    # Eh = masked_data/torch.repeat_interleave(torch.mean(masked_data[:,:,:nbase],dim=2).unsqueeze_(2),sig.shape[2],dim=2)
    Eh = Eh * mask_t
    for x in range(cnt_row):
        for y in range(cnt_col):
            out_patches[:, cnt, :, :] = Eh[:, :, UL_row + (x - 1):UL_row + (x - 1) + patch_size,
                                        UL_col + (y - 1):UL_col + (y - 1) + patch_size].squeeze()
            cnt += 1

    out_patches = out_patches.permute(2, 3, 0, 1)
    out_patches = out_patches.view(patch_size ** 2, Eh.shape[1], 1, -1)
    return out_patches


def assemble_sliding_patches(patch_size, patches, mask=None):
    if mask != None:
        cord = (mask[0, 0, :, :] == 1).nonzero(as_tuple=True)
        hw = patch_size // 2
        UL_row = torch.min(cord[0]) - hw
        if UL_row == 0:
            UL_row = 1
        UL_col = torch.min(cord[1]) - hw
        LR_row = torch.max(cord[0]) + hw
        LR_col = torch.max(cord[1]) + hw
        mask_t = torch.repeat_interleave(mask, patches.shape[1], dim=0)
    else:
        cord = (torch.arange(320), torch.arange(320))
        hw = patch_size // 2
        UL_row = torch.min(cord[0])
        UL_col = torch.min(cord[1])
        LR_row = torch.max(cord[0])
        LR_col = torch.max(cord[1])

    cnt_row = LR_row - UL_row - 1
    cnt_col = LR_col - UL_col - 1
    cnt = 0
    temp_patch = torch.zeros(patches.shape[1], 1, mask.shape[2], mask.shape[3])
    for x in range(cnt_row):
        for y in range(cnt_col):
            temp_patch[:, :, UL_row + x, UL_col + y] = patches[cnt, :].unsqueeze_(1)
            cnt += 1
    temp_patch = temp_patch * mask_t
    # print('outpatch_cif =',temp_patch.shape)
    # temp_patch = temp_patch.permute(2,3,0,1).squeeze()
    return temp_patch


sublist = ['sub3','sub5','sub6','sub7','sub8','sub9','sub10','sub11','sub12','sub14','sub15','sub17','sub18']
# sublist = ['sub9','sub10','sub11','sub12','sub14','sub15','sub16','sub17','sub18']
sublist = ['sub3']
# sublist = ['sub6','sub7','sub8','sub9','sub10','sub11','sub12','sub14','sub15','sub17','sub18']


for c in sublist:
    caseID = c
    print(caseID)
    basePath = 'F:\\Research\\GRASP_BBB\\' + caseID + '\\'
    fName = 'F:\\Research\\GRASP_BBB\\' + caseID + '\\' + caseID + '_cts_MC.mat'
    fName = 'F:\\Research\\GRASP_BBB\\' + caseID + '\\' + caseID + '_cts_MC_gf3.mat'


    f = io.loadmat(fName)
    # cts_nufft = torch.from_numpy(np.array(f['cts_nufft'])).unsqueeze(0).transpose(2,3).transpose(1,2)
    cts_grasp = torch.from_numpy(np.array(f['cts_grasp'])).transpose(2,0).transpose(3,1)
    # sl = int(torch.from_numpy(f['sl_sel']).squeeze().detach().numpy())


    fName = 'F:\\Research\\GRASP_BBB\\' + caseID + '\\' + caseID + '_T10_B1_All.mat'
    f = io.loadmat(fName)
    brain_mask = torch.from_numpy(np.array(f['mask_128']))

    model = VT_113()
    model.to(device)
    state_dict = torch.load('DCE_Vit_small', map_location=device)
    # create new OrderedDict that does not contain `module.`

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict['model_state_dict'].items():
        new_state_dict[k] = v
    # load params
    model.load_state_dict(new_state_dict)
    Cp_grasp = np.zeros((128,128,20,113))
    for sl in range(20):
        print(sl)
        out = gen_sliding_patches(3,cts_grasp[sl,:,:,:].unsqueeze(0),10,brain_mask[:,:,sl].unsqueeze(0).unsqueeze(0))
        out = out.permute(3,2,0,1)
        out = Variable(out)
        out = out.float()
        temp = model(out)
        temp = assemble_sliding_patches(3,temp,brain_mask[:,:,sl].unsqueeze(0).unsqueeze(0)).permute(2,3,0,1).squeeze()
        Cp_grasp[...,sl,:] = temp.detach().numpy()

    # out = gen_sliding_patches(3, cts_nufft, 10,
    #                           brain_mask[:, :, sl].unsqueeze(0).unsqueeze(0))
    # out = out.permute(3, 2, 0, 1)
    # out = Variable(out)
    # out = out.float()
    # temp = model(out)
    # temp = assemble_sliding_patches(3, temp, brain_mask[:, :, sl].unsqueeze(0).unsqueeze(0)).permute(2, 3, 0,
    #                                                                                                  1).squeeze()
    # Cp_nufft = temp.detach().numpy()

    fOut = basePath + caseID + '_Cp_MC_biexp2.mat'
    # fOut = basePath + caseID + '_Cp_MC.mat'

    # io.savemat(fOut, {'Cp_grasp':Cp_grasp,'Cp_nufft':Cp_nufft})
    io.savemat(fOut, {'Cp_grasp': Cp_grasp})
    # io.savemat(fOut, {'Cp_nufft': Cp_nufft})

'''
Valid_X = Variable(X)
Valid_X = Valid_X.float()


pred_X = model(Valid_X)
pred_X = pred_X.detach().numpy()


io.savemat('Patch_Result.mat',{'result':pred_X.squeeze()})

'''