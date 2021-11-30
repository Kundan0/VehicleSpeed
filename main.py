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
            self.samples.append((inter_tensor,velocity,position))

            break
    def __len__(self):
        return len(self.samples)  

    def __getitem__(self, index) :
        return self.samples[index]

class myModel(nn.Module):
    #hl_dims are hidden layer dimensions
    def __init__(self,input_dims,hl_dim1,hl_dim2,batch_size=1,optim=torch.optim.Adam,output_dims=4):
        super.__init__()
        self.input_dims=input_dims
        self.output_dims=output_dims
        self.batch_size=batch_size
        self.hl_dim1=hl_dim1
        self.hl_dim2=hl_dim2
        self.fc1=nn.Linear(self.input_dims,self.hl_dim1)
        self.fc2=nn.Linear(self.hl_dim1,self.hl_dim2)
        self.fc3=nn.Linear(self.hl_dim2,self.output_dims)

    def forward(self,data):
        flattened_data=torch.flatten(data)
        x=self.fc1(data)
        x=self.fc2(x)
        result=self.fc3(x)
        return result


    def training_step(self,batch):
        result=self(batch)
        loss=torch.




    

    def evaluate(self):
        pass  
    
    
        

folder_dir=os.path.expanduser('~')+'/gogly/Depth2/All/'
datas=myDataset(folder_dir,(240,320))
print(datas.samples[0][0].size(),datas.samples[0][1],datas.samples[0][2])
#train_dl=DataLoader(datas,batch_size=1,shuffle=True)
model=myModel(1,2,3)

print(model)

