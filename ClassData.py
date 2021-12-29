import torch
import os
import json
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms.transforms import ToPILImage

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
        top=int(annotation_data["bbox"]["top"]/HEIGHT_RATIO)
        left=int(annotation_data["bbox"]["left"]/WIDTH_RATIO)
        right=int(annotation_data["bbox"]["right"]/WIDTH_RATIO)
        bottom=int(annotation_data["bbox"]["bottom"]/HEIGHT_RATIO)
        
        label=torch.cat((velocity,position),dim=0)
        
        
        PATH_Depth=os.path.join(self.depth_dir,folder,"imgs")
        PATH_OF=os.path.join(self.of_dir,folder)
        
        depth_tensor1=ToTensor()(Image.open(os.path.join(PATH_Depth,filename1)).crop((0,50,320,240)).resize(self.size))[0]
        
        depth_tensor2=ToTensor()(Image.open(os.path.join(PATH_Depth,filename2)).crop((0,50,320,240)).resize(self.size))[0]

        of_tensor1=ToTensor()((Image.open(os.path.join(PATH_OF,filename1.lstrip('0').replace('.jpg','')+"a.png")).crop((0,200,1280,720)).resize(self.size)))[0]
        of_tensor2=ToTensor()((Image.open(os.path.join(PATH_OF,filename1.lstrip('0').replace('.jpg','')+"b.png")).crop((0,200,1280,720)).resize(self.size)))[0]

        

        DELTA=20
        HALF_DELTA=int(DELTA/2)
        
        bbox_mask=torch.zeros(self.size[::-1])
        
        
        
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


        
        

        return (inter_tensor.permute(0,2,1),label)              
        
