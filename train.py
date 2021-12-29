import torch
import os
from torch.utils.data import DataLoader
from ClassData import myDataset
from ClassModel import myModel
from DeviceData import DeviceDataLoader
import numpy as np

PATH=os.curdir

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
print(model_name)
device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
print(device)    
depth_dir=os.path.join(PATH,"Depth2")
of_dir=os.path.join(PATH,"NewOpticalFlow")
an_dir=os.path.join(PATH,"Annotations")
json_dir=os.path.join(PATH,json_name)

dataset=myDataset(json_dir,depth_dir,of_dir,an_dir)
train_dl=DataLoader(dataset,batch_size=64,shuffle=False)

train_dl_device=DeviceDataLoader(train_dl,device)

model=myModel().to(device)

def fit(epochs,model,train_dl_device,learning_rate,optim=torch.optim.SGD):
    optimizer=optim(model.parameters(),learning_rate)
    
    
    try:
        model.load_state_dict(torch.load(os.path.join(PATH,"State",model_name)))
        print("Loading Model ...")
    except:
        print("Cannot Load")

    for ep in range(epochs):
        print("epoch",ep)
        history=[]
        for idx,batch in enumerate(train_dl_device):
            print("idx",idx)
            optimizer.zero_grad()
            loss=model.training_step(batch)
            l=loss.detach().cpu().item()
            print("loss",l)
            loss.backward()
            optimizer.step() 
            if idx % 50 ==0 and idx!=0:
                
                torch.save(model.state_dict(),os.path.join(PATH,"State",model_name))
                print("saving model")
            history.append(l)
            print("average_Loss for last 50 batches",np.average(history[-50:]))
        
            
            
        
        

fit(10,model,train_dl_device,0.001,torch.optim.Adam)




