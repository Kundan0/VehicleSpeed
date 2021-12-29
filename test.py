import torch

import torch.nn as nn
import os
import json
from torch.utils.data import DataLoader,Dataset
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms.transforms import ToPILImage


PATH=os.curdir

class myDataset(Dataset):
    def __init__(self,json_dir,depth_dir,of_dir,annotation_dir,size=(128,72)):
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
        HEIGHT_RATIO=10
        WIDTH_RATIO=10
        top=int(annotation_data["bbox"]["top"])
        left=int(annotation_data["bbox"]["left"])
        right=int(annotation_data["bbox"]["right"])
        bottom=int(annotation_data["bbox"]["bottom"])
        
        label=torch.cat((velocity,position),dim=0)
        
        
        PATH_Depth=os.path.join(self.depth_dir,folder,"imgs")
        PATH_OF=os.path.join(self.of_dir,folder)
        
        CROP_SIZE_O=200
        CROP_SIZE_D=int(CROP_SIZE_O/3)

        depth_tensor1=ToTensor()(Image.open(os.path.join(PATH_Depth,filename1)).crop((0,CROP_SIZE_D,320,240)).resize(self.size))[0]
        
        depth_tensor2=ToTensor()(Image.open(os.path.join(PATH_Depth,filename2)).crop((0,CROP_SIZE_D,320,240)).resize(self.size))[0]

        of_tensor1=ToTensor()((Image.open(os.path.join(PATH_OF,filename1.lstrip('0').replace('.jpg','')+"a.png")).crop((0,CROP_SIZE_O,1280,720)).resize(self.size)))[0]
        of_tensor2=ToTensor()((Image.open(os.path.join(PATH_OF,filename1.lstrip('0').replace('.jpg','')+"b.png")).crop((0,CROP_SIZE_O,1280,720)).resize(self.size)))[0]

        

        DELTA=10
        HALF_DELTA=int(DELTA/2)
        
        bbox_mask=torch.zeros(self.size[::-1])
        
        
        top=int(((top/3)-CROP_SIZE_D)/((240-CROP_SIZE_D)/self.size[1]))
        bottom=int(((bottom/3)-CROP_SIZE_D)/((240-CROP_SIZE_D)/self.size[1]))
        
        left=int(left/WIDTH_RATIO)
        right=int(right/WIDTH_RATIO)
        
        try:
            bbox_size=(bottom-top+DELTA,right-left+DELTA)
            ones=torch.ones(bbox_size)
            bbox_mask[top-HALF_DELTA:bottom+HALF_DELTA,left-HALF_DELTA:right+HALF_DELTA]=ones
        except:
            bbox_size=(bottom-top,right-left)
            ones=torch.ones(bbox_size)
            bbox_mask[top:bottom,left:right]=ones
        
        
        
        inter_tensor=torch.cat((depth_tensor1.unsqueeze(0),of_tensor1.unsqueeze(0)),dim=0)
        inter_tensor=torch.cat((inter_tensor,of_tensor2.unsqueeze(0)),dim=0)
        inter_tensor=torch.cat((inter_tensor,depth_tensor2.unsqueeze(0)),dim=0)
        inter_tensor=torch.cat((inter_tensor,bbox_mask.unsqueeze(0)),dim=0)


        # depth_image1=ToPILImage()(depth_tensor1)
        # depth_image2=ToPILImage()(depth_tensor2)
        # bbox_mask_image=ToPILImage()(bbox_mask)
        # of_tensorimage1=ToPILImage()(of_tensor1)
        # of_tensorimage2=ToPILImage()(of_tensor2)
        # depth_image1.show()
        # depth_image2.show()
        # bbox_mask_image.show()
        # of_tensorimage1.show()
        # of_tensorimage2.show()
        

        return (inter_tensor.permute(0,2,1),label)              
        



class myModel(nn.Module):
    #hl_dims are hidden layer dimensions
    def __init__(self,size=(128,72),hl_dim1=512,hl_dim2=256,hl_dim3=128,output_dims=4,loss=nn.MSELoss()):
        super().__init__()
        
        self.output_dims=output_dims
        self.size=size
        self.hl_dim1=hl_dim1
        self.hl_dim2=hl_dim2
        self.hl_dim3=hl_dim3

        
        self.loss=loss
        self.n1=nn.Sequential(
            
            nn.Linear(5*self.size[0]*self.size[1],self.output_dims),
            

        )
        

    def forward(self,image):
        inte=torch.flatten(image,start_dim=1)
        
        result=self.n1(inte)
        
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
#device=torch.device('cpu')
learn_type=2
if learn_type==1:
    model_name="trained1"
    json_name="JSONa.json"
elif learn_type==2:
    model_name="trained2"
    json_name="JSONb.json"
else:
    model_name="trained3"
    json_name="JSONc.json"



depth_dir=os.path.join(PATH,"Depth2")
of_dir=os.path.join(PATH,"NewOpticalFlow")
an_dir=os.path.join(PATH,"Annotations")
json_dir=os.path.join(PATH,json_name)
dataset=myDataset(json_dir,depth_dir,of_dir,an_dir)


train_dl=DataLoader(dataset,batch_size=32,shuffle=False)

train_dl_device=DeviceDataLoader(train_dl,device)

model=myModel().to(device)
history=[]
def fit(epochs,model,train_dl_device,learning_rate,optim=torch.optim.SGD):
    optimizer=optim(model.parameters(),learning_rate)
    
    try:
        model.load_state_dict(torch.load(os.path.join(PATH,"State",model_name)))
        print("hel")
    except:
        print("cannot load")
    for ep in range(epochs):
        print("epoch",ep)
        for idx,batch in enumerate(train_dl_device):
            print("idx",idx)
            optimizer.zero_grad()
            loss=model.training_step(batch)
            print("loss",loss.detach().cpu().item())
            loss.backward()
            optimizer.step() 
            if idx % 100 ==0 and idx!=0:
                torch.save(model.state_dict(),os.path.join(PATH,"State",model_name))
                
            
        
            
            
            
        
        #torch.save(model.state_dict(),os.path.join(os.curdir,"State\model"))

fit(100,model,train_dl_device,0.001,torch.optim.Adam)
#fit(1000,model,train_dl,0.001,torch.optim.Adam)



