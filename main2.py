import torch
import torch.nn as nn
import os
import json
from torch.nn.modules.linear import Linear
from torch.utils.data import DataLoader,Dataset,random_split
from matplotlib.image import imread
import numpy as np
import torch.functional as F
from PIL import Image
from torchvision.transforms import GaussianBlur,ToTensor,Compose,Resize

def choose(item):
    return int(item)

class myDataset(Dataset):
    def __init__(self,json_dir,depth_dir,of_dir,annotation_dir,size=(512,288)):
        super().__init__()
        self.depth_dir=depth_dir
        self.of_dir=of_dir
        self.annotation_dir=annotation_dir
        self.json_dir=json_dir
        
        self.size=size
    
    
    
    def __len__(self):
        return  len(json.load(open(self.json_dir)))

    def __getitem__(self, index) :
        
        json_data=json.load(open(self.json_dir))[index]
        folder=json_data["folder"]
        filename1=json_data["image_name"]
        filename2=str(int(filename1.replace('.jpg',''))+1).zfill(3)+".jpg"
        annotation_index=json_data["an_index"]
        annotation_data=json.load(open(os.path.join(self.annotation_dir,"annotation"+folder+".json")))[annotation_index]


        velocity=torch.tensor(annotation_data['velocity'])
        
        position=torch.tensor(annotation_data["position"])
        HEIGHT_RATIO=2.5
        WIDTH_RATIO=2.5
        top=int(annotation_data["bbox"]["top"]/HEIGHT_RATIO)
        left=int(annotation_data["bbox"]["left"]/WIDTH_RATIO)
        right=int(annotation_data["bbox"]["right"]/WIDTH_RATIO)
        bottom=int(annotation_data["bbox"]["bottom"]/HEIGHT_RATIO)
        
        label=torch.cat((velocity,position),dim=0)
        
        #PATH=self.depth_dir+"\\"+folder+"\\imgs"
        PATH=os.path.join(self.depth_dir,folder,"imgs")
        PATH_OF=os.path.join(self.of_dir,folder)
        # depth_tensor1=ToTensor()((Image.open(PATH+"\\"+filename1)[:,:,0]).resize(self.size)).permute(1,0)
        
        # depth_tensor2=ToTensor()(Image.open(PATH+"\\"+filename2)[:,:,0]).resize(self.size).permute(1,0)

        # of_tensor1=ToTensor()((Image.open(PATH+"\\"+filename1.lstrip('0').replace('.jpg','')+"a.png")[:,:,0]).resize(self.size)).permute(1,0)
        # of_tensor2=ToTensor()(Image.open(PATH+"\\"+filename1.lstrip('0').replace('.jpg','')+"b.png")[:,:,0]).resize(self.size).permute(1,0)
        depth_tensor1=ToTensor()(Image.open(os.path.join(PATH,filename1).resize(self.size)))[0].permute(1,0)
        
        depth_tensor2=ToTensor()(Image.open(os.path.join(PATH,filename2).resize(self.size)))[0].permute(1,0)

        of_tensor1=ToTensor()((Image.open(os.path.join(PATH_OF,filename1.lstrip('0').replace('.jpg','')+"a.png").resize(self.size))))[0].permute(1,0)
        of_tensor2=ToTensor()((Image.open(os.path.join(PATH_OF,filename1.lstrip('0').replace('.jpg','')+"b.png").resize(self.size))))[0].permute(1,0)

        
        DELTA=20
        HALF_DELTA=int(DELTA/2)
        
        bbox_mask=torch.zeros(self.size[::-1])
        bbox_size=(bottom-top+DELTA,right-left+DELTA)
        ones=torch.ones(bbox_size)
        bbox_mask[top-HALF_DELTA:bottom+HALF_DELTA,left-HALF_DELTA:right+HALF_DELTA]=ones
        
        
        

        inter_tensor=torch.cat((depth_tensor1.unsqueeze(0),of_tensor1.unsqueeze(0)),dim=0)
        inter_tensor=torch.cat((inter_tensor,of_tensor2.unsqueeze(0)),dim=0)
        inter_tensor=torch.cat((inter_tensor,depth_tensor2.unsqueeze(0)),dim=0)
        inter_tensor=torch.cat((inter_tensor,bbox_mask.permute(1,0).unsqueeze(0)),dim=0)

        

        return (inter_tensor,label)              
        



class myModel(nn.Module):
    #hl_dims are hidden layer dimensions
    def __init__(self,size=(512,288),hl_dim1=1024,hl_dim2=512,hl_dim3=256,output_dims=4,loss=nn.MSELoss()):
        super().__init__()
        
        self.output_dims=output_dims
        self.size=size
        self.hl_dim1=hl_dim1
        self.hl_dim2=hl_dim2
        self.hl_dim3=hl_dim3

        self.network=nn.Sequential(

            nn.Flatten(),
            nn.Linear(5*self.size[0]*self.size[1],self.hl_dim1),
            nn.Linear(self.hl_dim1,self.hl_dim2),
            nn.Linear(self.hl_dim2,self.hl_dim3),
            nn.Linear(self.hl_dim3,self.output_dims)
            


        )
        self.loss=loss

    def forward(self,image):
       
        
        result=self.network(image)
       
        return result
        


    def training_step(self,batch):
        image,label=batch
        result=self(image)
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


device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')        

depth_dir=os.path.join(os.curdir,"Depth2")
of_dir=os.path.join(os.curdir,"FullOF")
an_dir=os.path.join(os.curdir,"Annotations")
json_dir=os.path.join(os.curdir,"JSON.json")
dataset=myDataset(json_dir,depth_dir,of_dir,an_dir)


train_dl=DataLoader(dataset,batch_size=1,shuffle=True)

train_dl_device=DeviceDataLoader(train_dl,device)

model=myModel().to(device)
history=[]
def fit(epochs,model,train_dl_device,learning_rate,optim=torch.optim.SGD):
    optimizer=optim(model.parameters(),learning_rate)
    
    for epoch in range(epochs):
        for batch in train_dl_device:
            
            optimizer.zero_grad()
            loss=model.training_step(batch)
            print("loss",loss.detach().cpu().item())
            loss.backward()
            optimizer.step()
            
            
            
        
        #torch.save(model.state_dict(),os.path.join(os.curdir,"State\model"))

fit(10,model,train_dl_device,0.001)


