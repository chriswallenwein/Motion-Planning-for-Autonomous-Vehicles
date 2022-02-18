import numpy as np
import os
from commonroad_geometric_io.common.io import list_files
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch
from torch.nn import SmoothL1Loss, BCELoss
from net import TrafficRepresentationNet

def train(data_loader, model, optimizer, epoch, loss_fn_cla, loss_fn_reg):
    """"
    Train a single epoch
    """
    train_loss = []
    for idx, data in enumerate(data_loader):
        optimizer.zero_grad()
        x=data.x.cuda()   

        edge_index = data.edge_index.cuda()
        edge_attr = data.edge_attr.cuda()
        y=data.y.cuda()
      
        cls_out, reg_out, en_cls_out, en_reg_out = model(x,edge_index,edge_attr)
        
        y_occupied_idx = y.gt(0.0)
        loss_cla = loss_fn_cla(cls_out, y_occupied_idx.float())
        loss_reg = loss_fn_reg(reg_out[y_occupied_idx], y[y_occupied_idx])
       
        loss = loss_cla + loss_reg*100
       
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        
        if (idx + 1) % 500 == 0:
            print("Train epoch: %d, iter: %d, loss_cla: %f" % (epoch, idx, loss_cla))
            print("Train epoch: %d, iter: %d, reg_loss: %f" % (epoch, idx, loss_reg*100))

    return np.mean(train_loss)


def val(data_loader, model, epoch, loss_fn_cla, loss_fn_reg):
    """
    Test(or evaluaction) function 
    """
    val_loss = []
    for idx, data in enumerate(data_loader):
        x=data.x.cuda()   
        edge_index = data.edge_index.cuda()
        edge_attr = data.edge_attr.cuda()
        y=data.y.cuda()
       
        cls_out, reg_out, en_cls_out, en_reg_out = model(x,edge_index,edge_attr)#.to(device)
        
        y_occupied_idx = y.gt(0.0)
        loss_cla = loss_fn_cla(cls_out, y_occupied_idx.float())
        loss_reg = loss_fn_reg(reg_out[y_occupied_idx], y[y_occupied_idx])
       
        loss = loss_cla + loss_reg*100
        
        val_loss.append(loss.item())
      
    return np.mean(val_loss)

def train_model(epochs, train_loader, test_loader, model_f, optimizer, loss_cla, loss_reg, lr_scheduler):
    """
    Training process
    """
    for epoch in range(epochs):   
            model_f.train()
            train_loss = train(train_loader, model_f, optimizer, epoch, loss_cla, loss_reg)

            model_f.eval()
            val_loss = val(test_loader, model_f, epoch, loss_fn_cla=loss_cla, loss_fn_reg=loss_reg)
            lr_scheduler.step()
            print("Train epoch: %d, lr: %f, loss: %f" % (epoch, lr_scheduler.get_lr()[0], train_loss))
            print("Test epoch: %d, loss: %f" % (epoch, val_loss))


def create_dataloader(dataset_path: str, mb_size: int) -> DataLoader:
            original_loader = torch.load(dataset_path)
            loader = DataLoader(
                original_loader,
                batch_size=mb_size,
                shuffle=True
            )
            return loader

if __name__ == "__main__":
    """
    An example of reconstruction model training
    """
    #change the data_dir to the path you storge data
    data_dir = '/content/drive/MyDrive/Autonomous Driving/code/commonroad-geometric-io/dataset'
    dataset_paths = list_files(data_dir, file_type='pth', join_paths=True)

    train_loader = create_dataloader(dataset_paths[-1], 64) #dataset_paths[-1] training data path
    test_loader = create_dataloader(dataset_paths[1], 64) #dataset_paths[1] test data path

    model_f = TrafficRepresentationNet(in_node=2,in_edge=3,out_channels=8,num_heads=1,out_conv=8) 
    loss_cla = BCELoss()
    loss_reg = SmoothL1Loss()
    model_f = model_f.cuda()

    optimizer = torch.optim.Adam(model_f.parameters(), lr=0.001, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,0.96, last_epoch=-1)
    train_model(50, train_loader, test_loader, model_f, optimizer, loss_cla, loss_reg, lr_scheduler)