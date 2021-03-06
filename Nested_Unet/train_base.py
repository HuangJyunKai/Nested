#For road segmentation
import torch
import argparse
import os
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.utils import save_image
from torch import nn, optim
from torchvision.transforms import transforms
from torch.optim import lr_scheduler
from nested_unet import  NestedWNet , UNet,NestedWNet_v2,NestedWNet_v3,NestedUNet,NestedUNet_RO
from nasunet import NasUNet,Base,NasUnetV1,BaseDownSample,SearchDownSample,DownSample_Decode_V1
from dataset_road import RoadDataset
from IOUEval import iouEval
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import cv2

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_main = transforms.Compose([
        transforms.Resize((256,512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        #
    ])
x_scale1 = transforms.Compose([
        transforms.Resize((768,1536)), 
        #transforms.RandomCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        #
    ])
x_scale2 = transforms.Compose([
        transforms.Resize((720,1280)), 
        #transforms.RandomCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        #
    ])
x_scale3 = transforms.Compose([
        transforms.Resize((512,1024)), 
        #transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        #
    ])
x_scale4 = transforms.Compose([
        transforms.Resize((384, 768)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        #
    ])
x_val = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        #
    ])
#y_transforms = transforms.ToTensor()
y_main = transforms.Compose([
    transforms.Resize((256,512)),]) 
y_scale1 = transforms.Compose([
    transforms.Resize((768,1536)),
])
y_scale2 = transforms.Compose([
    transforms.Resize((720,1280)),
])
y_scale3 = transforms.Compose([
    transforms.Resize((512,1024)),
])
y_scale4 = transforms.Compose([
    transforms.Resize((384,768)),
])
y_val = transforms.Compose([
    transforms.Resize((256,512)),
])
nclass=3 # 0 background
IGNORE_LABEL = 0
def validation(epoch,model, criterion, optimizer, val_loader):
    iouEvalVal = iouEval(nclass)
    model.eval()
    step=0
    epoch_loss = 0
    epoch_mIOU = 0
    epoch_acc = 0. 
    dt_size = len(val_loader.dataset)
    with torch.no_grad():
        for x, y in val_loader:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            # forward
            #outputs = model(inputs)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            print("%d/%d,val_loss:%0.5f " % (step, len(val_loader), loss.item()))
            #mIOU
            output = torch.softmax(outputs,dim=1)
            iouEvalVal.addBatch(output.max(1)[1].data, labels.data)
        overall_acc, per_class_acc, per_class_iou, mIOU = iouEvalVal.getMetric()
    print("epoch %d val_loss:%0.5f " % (epoch+1, epoch_loss/step))
    print("overall_acc :",overall_acc)
    print("per_class_acc :",per_class_acc)
    print("per_class_iou :",per_class_iou)
    print("mIOU :",mIOU)
    return epoch_loss/step , overall_acc, per_class_acc, per_class_iou, mIOU


def train_model(model, criterion, optimizer, train_loader, scheduler, epoch, num_epochs):
    iouEvalTrain = iouEval(nclass)
    #scheduler.step()
    dt_size = len(train_loader.dataset)
    epoch_loss = 0
    step = 0
    #num_correct = 0
    for x, y in train_loader:
        num_correct = 0
        step += 1
        inputs = x.to(device)
        labels = y.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        #outputs = model(inputs)
        outputs= model(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
            
        output = torch.softmax(outputs,dim=1)
        iouEvalTrain.addBatch(output.max(1)[1].data, labels.data)
            
        print("%d/%d,train_loss:%0.5f " % (step, len(train_loader), loss))
        ####################################################
        #print (model.arch_parameters ()) #for branch search
        ####################################################
    overall_acc, per_class_acc, per_class_iou, mIOU = iouEvalTrain.getMetric()
    print("overall_acc :",overall_acc)
    print("per_class_acc :",per_class_acc)
    print("per_class_iou :",per_class_iou)
    print("mIOU :",mIOU)
    dirName = "./models/road_BaseDownSample_150ep/"
    if not os.path.exists(dirName):
       os.mkdir(dirName)
       print("Directory " , dirName ,  " Created ")  
    if epoch%5==4:
        torch.save(model.state_dict(), dirName+'road_BaseDownSample_150ep_weights_epoch_%d.pth' % (epoch+1))
    return epoch_loss/step, overall_acc, per_class_acc, per_class_iou, mIOU

#训练模型
def train(args):
    num_epochs = 150
    step_size  = 50
    gamma      = 0.5
    model = BaseDownSample(3,3).to(device)
    
    batch_size = args.batch_size
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs
    #criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL,reduction='mean')
    criterion = nn.CrossEntropyLoss()
    road_dataset= RoadDataset("../YeeDragon/data/training/",transform=x_main,target_transform=y_main)
    road_dataset_scale1 = RoadDataset("../YeeDragon/data/training/",transform=x_scale1,target_transform=y_scale1)
    road_dataset_scale2 = RoadDataset("../YeeDragon/data/training/",transform=x_scale2,target_transform=y_scale2)
    road_dataset_scale3 = RoadDataset("../YeeDragon/data/training/",transform=x_scale3,target_transform=y_scale3)
    road_dataset_scale4 = RoadDataset("../YeeDragon/data/training/",transform=x_scale4,target_transform=y_scale4)
    road_dataset_val    = RoadDataset("../YeeDragon/data/training/",transform=x_val,target_transform=y_val)
    #split training and validation
    validation_split = .2
    shuffle_dataset = True
    dataset_size = len(road_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(38)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    #multi-scale dataloader
    train_loader = DataLoader(road_dataset, batch_size=batch_size, sampler=train_sampler)
    train_loaderscale1 = DataLoader(road_dataset_scale1, batch_size=batch_size, sampler=train_sampler)
    train_loaderscale2 = DataLoader(road_dataset_scale2, batch_size=batch_size, sampler=train_sampler)
    train_loaderscale3 = DataLoader(road_dataset_scale3, batch_size=batch_size, sampler=train_sampler)
    train_loaderscale4 = DataLoader(road_dataset_scale4, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(road_dataset_val, batch_size=batch_size,sampler=valid_sampler)
    
    for epoch in range(num_epochs):
        print("epoch: %d/%d" %(epoch+1,num_epochs)) 
        #print("scale : 768x1536")
        #train_model(model, criterion, optimizer, train_loaderscale1, scheduler, epoch, num_epochs)
        #print("scale : 720x1280")
        #train_model(model, criterion, optimizer, train_loaderscale2, scheduler, epoch, num_epochs)
        #print("scale : 512x1024")
        #train_model(model, criterion, optimizer, train_loaderscale3, scheduler, epoch, num_epochs)
        #print("scale : 384x768")
        #train_model(model, criterion, optimizer, train_loaderscale4, scheduler, epoch, num_epochs)
        print("scale : 256x512")
        epoch_loss, overall_acc, per_class_acc, per_class_iou, mIOU = train_model(model, criterion, optimizer, train_loader,scheduler, epoch, num_epochs)
        print("scale : 256x512")
        valepoch_loss,valoverall_acc, valper_class_acc, valper_class_iou, valmIOU = validation(epoch, model, criterion, optimizer, validation_loader)
        
        fp = open("road_BaseDownSample_150ep_epoch_%d.txt" % num_epochs, "a")
        fp.write("epoch %d train_loss:%0.3f \n" % (epoch+1, epoch_loss))
        fp.write("train overall_acc:%0.3f \n"%(overall_acc))
        fp.write("train per_class_acc: ")
        fp.write(str(per_class_acc))
        fp.write("\n")
        fp.write("train per_class_iou: ")
        fp.write(str(per_class_iou))
        fp.write("\n")
        fp.write("train mIOU:%0.3f\n"% (mIOU))
        fp.write("epoch %d val_loss:%0.3f \n" % (epoch+1, valepoch_loss))
        fp.write("val overall_acc:%0.3f \n" % (valoverall_acc))
        fp.write("val per_class_acc: ")
        fp.write(str(valper_class_acc))
        fp.write("\n")
        fp.write("vak per_class_iou: ")
        fp.write(str(valper_class_iou))
        fp.write("\n")
        fp.write("val mIOU:%0.3f\n"% (valmIOU))
        fp.write("\n")
        fp.close()

def Generate(args):
    model = ESPNet(12, p=2, q=3).to(device)
    model.load_state_dict(torch.load(args.ckpt,map_location='cpu'))
    model.eval()
    count=0
    root = './data/testing/'
    print("Generate...")
    import time
    tStart = time.time()
    for filename in os.listdir(root):
        if filename == '.ipynb_checkpoints':
            continue
        imgroot=os.path.join(root+filename)
        name = filename[:-4]
        print("Image Processing: ",name)
        img = Image.open(imgroot)
        img = x_transforms(img)
        img = img.view(1,3,256,512) #forreference
        #img = img.view(1,3,1080,1920) #forreference
        img = img.to(device)
        with torch.no_grad():
            output = model(img)
            output = torch.softmax(output,dim=1)
            N, _, h, w = output.shape
            pred = output.transpose(0, 2).transpose(3, 1).reshape(-1, 12).argmax(axis=1).reshape(N, h, w) #class 12
            pred = pred.squeeze(0)
            print(np.unique(pred.cpu()))
            Decode_image(pred,name)
    tEnd = time.time()#計時結束
    #列印結果
    print ("It cost %f sec" % (tEnd - tStart))#會自動做近位

def Decode_image(img_n,name):
    pallete = [[0,0,0] , 
            [255, 0, 255], [128, 0, 128], [0, 64, 64], [0, 0, 0], [0, 0, 0], 
            [255, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0], 
            [255, 0, 255]]
    img_n = img_n.cpu()
    img_ans=np.zeros((img_n.shape[0],img_n.shape[1],3), dtype=np.int) #class 12
    for idx in range(len(pallete)):
        [b, g, r] = pallete[idx]
        img_ans[img_n == idx] = [b, g, r] 
    im_ans = Image.fromarray(np.uint8(img_ans)).convert('RGB') 
    im_ans = cv2.cvtColor(np.array(im_ans),cv2.COLOR_RGB2BGR)         
    #return im_ans           
    cv2.imwrite("./Result/"+name+"_espnet_pred_30_nojitter.png",im_ans)
if __name__ == '__main__':
    #参数解析
    parse=argparse.ArgumentParser()
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=8)
    parse.add_argument("--ckpt", type=str, help="the path of model weight file")
    args = parse.parse_args()

    if args.action=="train":
        train(args)
    elif args.action=="test":
        test(args)
    elif args.action=="model":
        Model_visualization(args)
    elif args.action=="generate":
        Generate(args)
    elif args.action=="oonx":
        oonx(args)
    elif args.action=="check":
        check_label(args)
    elif args.action=="iou":
        Val(args)
        


