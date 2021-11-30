import torch
import torch.nn as nn
import os
import json
from torch.utils.data import DataLoader,Dataset
from matplotlib.image import imread
import numpy as np
import torch.functional as F


class myDataset(Dataset):
    def __init__(self,root_dir,image_size,transform=None):

        self.root_dir=root_dir
        self.transform=transform
        self.samples=[]
        self.image_size=image_size
        self.call_init()
        
    def call_init(self):
        for sub_directory in os.listdir(self.root_dir):
            annotation_data=json.load(open(self.root_dir+'/'+sub_directory+'/annotation.json'))
            velocity=torch.from_numpy(np.array(annotation_data[0]['velocity']))
            position=torch.from_numpy(np.array(annotation_data[0]["position"]))
            label=torch.cat((velocity,position),dim=0)
            dir_name=self.root_dir+'/'+sub_directory+'/imgs'
            inter_tensor=None
            for image in os.listdir(dir_name):

                image_name=dir_name+'/'+image
                img=imread(image_name)
                
                depth_tensor=torch.from_numpy(np.array(img[:,:,0]))
                #print('depth tensor sixe',depth_tensor.size())
                if inter_tensor!=None:

                    inter_tensor=torch.cat((inter_tensor,depth_tensor.unsqueeze(0)),dim=0)
                
                       
                else :
                    inter_tensor=depth_tensor.unsqueeze(0)
            self.samples.append((inter_tensor,label))

            break
    def __len__(self):
        return len(self.samples)  

    def __getitem__(self, index) :
        return self.samples[index]

class myModel(nn.Module):
    #hl_dims are hidden layer dimensions
    def __init__(self,input_dims,hl_dim1,hl_dim2,batch_size=1,output_dims=4,loss=nn.MSELoss):
        super.__init__()
        self.input_dims=input_dims
        self.output_dims=output_dims
        self.batch_size=batch_size
        self.hl_dim1=hl_dim1
        self.hl_dim2=hl_dim2
        self.fc1=nn.Linear(self.input_dims,self.hl_dim1)
        self.fc2=nn.Linear(self.hl_dim1,self.hl_dim2)
        self.fc3=nn.Linear(self.hl_dim2,self.output_dims)
        self.loss=loss

    def forward(self,data):
        flattened_data=torch.flatten(data)
        x=self.fc1(data)
        x=self.fc2(x)
        result=self.fc3(x)
        return result


    def training_step(self,batch):
        input,label=batch
        result=self(input)
        loss=self.loss(result,label)
        return loss
    





    

    def evaluate(self):
        pass  
    
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        
        return [to_device(x, device) for x in data]

    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


device= torch.device('cuda') if t.cuda.is_available() else torch.device('cpu')        

folder_dir=os.path.expanduser('~')+'/gogly/Depth2/All/'
dataset=myDataset(folder_dir,(240,320))
train_dl=DataLoader(dataset,batch_size=1,shuffle=True,num_workers=2)
train_dl=DeviceDataLoader(train_dl,device)
model=myModel()


def fit(epochs,model,train_dl,learning_rate,optim=torch.optim.SGD):
    optimizer=torch.optim.Adam(model.parameters(),learning_rate)
    history=[]
    for epoch in range(epochs):
        for batch in train_dl:
            loss=model.training_step(batch)
            loss.backward()
        optimizer.step()


print(model)


