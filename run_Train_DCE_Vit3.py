import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import lr_scheduler
import time
from scipy import io
import matplotlib.pyplot as plt
import h5py
# from DCE_Vit import VisionTransformer
from DCE_Vit_small_nt_113 import VisionTransformer
"""

DCE_Vit : 100,000 TrainData, batch_size=50
DCE_Vit2 : 1million TrainData, batch_size=5
DCE_Vit3 : 1million TrainData, batch_size=5, patch in time dim.



"""




class DCE_Train_ImgDataset(Dataset):
    def __init__(self):
        with h5py.File('/home/hpc/rzku/mlvl116h/Python_Code/New Train_Data/Tumor_testData_10.mat', 'r') as f:
        # with h5py.File('TrainingData_TCM.mat', 'r') as f:
            self.X = np.array(f['Train_Data'])
            self.X = torch.from_numpy(np.moveaxis(self.X, [0, 1, 2, 3], [3,2,1,0]))

            self.y = np.array(f['Train_CIFs'])
            #self.y = torch.from_numpy(np.moveaxis(self.y, [0, 1, 2], [-1, -2, -3]))

            #self.y = self.y[:,:,:,:1000]

        self.len = self.X.shape[-1]
        print(self.len)

    def __getitem__(self, index):
        return self.X[:,:,:,index], self.y[:, index]

    def __len__(self):
        return self.len

class DCE_Valid_ImgDataset(Dataset):
    def __init__(self):
        with h5py.File('/home/hpc/rzku/mlvl116h/Python_Code/New Train_Data/Tumor_ValidData_10.mat', 'r') as f:
        # with h5py.File('ValidData_TCM.mat', 'r') as f:
            self.X = np.array(f['Valid_Data'])
            self.X = torch.from_numpy(np.moveaxis(self.X, [0, 1, 2, 3], [3, 2, 1, 0]))

            self.y = np.array(f['Valid_CIFs'])
            # self.y = torch.from_numpy(np.moveaxis(self.y, [0, 1, 2], [-1, -2, -3]))

            # self.y = self.y[:,:,:,:1000]

        self.len = self.X.shape[-1]
        print(self.len)

    def __getitem__(self, index):
        return self.X[:, :, :, index], self.y[:, index]

    def __len__(self):
        return self.len

def main():
    start = time.time()

    patch_wise_cost = 0

    device = torch.device("cuda")
    batch_size = 5

    Train_data = DCE_Train_ImgDataset()
    train_loader = DataLoader(dataset=Train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    Valid_data = DCE_Valid_ImgDataset()
    valid_loader = DataLoader(dataset=Valid_data, batch_size=batch_size, shuffle=True, num_workers=2)
    Train_loss = []
    Valid_loss = []

    model = VisionTransformer()
    model.to(device)

    current_lr = 1

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.MSELoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.985)
    max_epoch = 1
    
    # checkpoint_interval = 40  # Save checkpoint every 40 epochs
    # start_epoch = 0

    # Load a previous checkpoint if it exists
    # checkpoint_file = 'DCE_Vit_1'
    
    '''if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])'''
    
    # for epoch in range(start_epoch, max_epoch):
    for epoch in range(max_epoch):
        tLoss = []
        for i, data in enumerate(train_loader, 0):
            optimizer.zero_grad()

            inputs, truePar = data

            inputs = inputs.permute(0, 3, 1, 2)

            batch_size = inputs.shape[0]

            inputs, truePar = Variable(inputs), Variable(truePar)
            inputs = inputs.to(device)
            truePar = truePar.to(device)

            y_pred = model(inputs.float())
            loss = criterion(y_pred.float().squeeze(), truePar.float().squeeze())

            print(epoch, i, loss.item())

            model.zero_grad()
            loss.backward()
            #        nn.utils.clip_grad_norm(model.parameters(), 1.)
            optimizer.step()
            tLoss.append(loss.item())
            
        torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'loss': loss,
                   'optimizer_state_dict': optimizer.state_dict()}, 'DCE_Vit_2')
        
        Train_loss.append(sum(tLoss) / len(tLoss))
        vLoss = []
        
        '''if epoch % checkpoint_interval == 0:
            checkpoint = {'epoch': epoch, 'loss': loss, 'model_state_dict': model.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict()}
        torch.save(checkpoint, checkpoint_file)'''
        
        for i, data in enumerate(valid_loader, 0):
            inputs, truePar = data

            inputs = inputs.permute(0, 3, 1, 2)

            batch_size = inputs.shape[0]

            inputs, truePar = Variable(inputs), Variable(truePar)
            inputs = inputs.to(device)
            truePar = truePar.to(device)

            y_pred = model(inputs.float())
            loss = criterion(y_pred.float().squeeze(), truePar.float().squeeze())

            vLoss.append(loss.item())

        Valid_loss.append(sum(tLoss) / len(tLoss))
        Last3 = min(Valid_loss[-3:])

        if (Valid_loss[-1] >= Last3) & (epoch > 100) & (current_lr > 1e-10):
            exp_lr_scheduler.step()
            print('Decreasing LR!')
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
                print("Current learning rate is: {}".format(current_lr))


        tLoss = []
        for i, data in enumerate(test_loader, 0):
            inputs, truePar = data

            inputs = inputs.permute(0, 3, 1, 2)

            batch_size = inputs.shape[0]

            inputs, truePar = Variable(inputs), Variable(truePar)
            inputs = inputs.to(device)
            truePar = truePar.to(device)

            y_pred = model(inputs.float())
            loss = criterion(y_pred.float().squeeze(), truePar.float().squeeze())

            tLoss.append(loss.item())
            if i == 0:
                pred_out = y_pred
                target_out = truePar
            else:
                pred_out = torch.cat((pred_out, y_pred), dim=0)
                target_out = torch.cat((target_out, truePar), dim=0)

        Test_loss.append(sum(tLoss) / len(tLoss))

        io.savemat('Testset_Result_Vit.mat', {'result': pred_out.detach().cpu().numpy().squeeze(),
                                          'GT': target_out.detach().cpu().numpy().squeeze(), 'TrainLoss': Train_loss,
                                          'ValidLoss': Valid_loss, 'TestLoss': Test_loss})

if __name__ == '__main__':
    main()





