import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import pandas as pd
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
from torch.utils.data import Dataset,DataLoader,TensorDataset
import matplotlib.pyplot as plt

#从文件夹路径读取图片数据并转换为张量
def images_to_tensor(folder_path):
    #定义图像转换操作
    trans=transforms.Compose([transforms.ToTensor()])
    image_tensors=[]
    #读取图像
    for file_path in sorted(os.listdir(folder_path),key=lambda x:int(x.split('.')[0])):  #os.listdir返回的是无序列表需手动排序
        #获取图像路径
        path=os.path.join(folder_path,file_path)
        #打开图片并转换为张量
        image=Image.open(path)
        image_tensor=trans(image)
        image_tensors.append(image_tensor)
    #将图像张量列表转换为张量数组
    ans=torch.stack(image_tensors)
    return ans

#从csv文件读取标签并顺序编码
def read_labels(csv_path):
    data=pd.read_csv(csv_path)
    labels=data['label']
    labels=LabelEncoder().fit_transform(labels)
    labels=torch.tensor(labels).to(torch.long)
    return labels.reshape(-1)

#创建train_iter并实时增广
def load_train_data(folder_path,csv_path,batch_size):
    #读取图像数据和标签
    images=images_to_tensor(folder_path)
    labels=read_labels(csv_path)
    #定义图像增广操作
    trans=transforms.Compose([transforms.ToPILImage(),
                              transforms.RandomHorizontalFlip(),
                              transforms.RandomVerticalFlip(),
                              transforms.RandomRotation(degrees=45),
                              transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                              transforms.Resize((224,224)),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # 创建自定义数据集类
    class MyDataset(Dataset):
        def __init__(self, images, labels, transform=None):
            self.images = images
            self.labels = labels
            self.transform = transform
        def __getitem__(self, index):
            image = self.images[index]
            label = self.labels[index]
            if self.transform:
                image = self.transform(image)
            return image, label
        def __len__(self):
            return len(self.images)
    dataset=MyDataset(images,labels,transform=trans)
    dataloader=DataLoader(dataset,batch_size=batch_size)
    return dataloader

#创建test_iter
def load_test_data(folder_path,batch_size):
    #读取图像数据
    images=images_to_tensor(folder_path)
    #定义图像增广操作
    trans=transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    #转换数据
    dataset=TensorDataset(images)
    class TransformedDataset(Dataset):
        def __init__(self, dataset, transform=None):
            self.dataset = dataset
            self.transform = transform
        def __getitem__(self, index):
            data = self.dataset[index][0]
            if self.transform:
                data = self.transform(data)
            return data
        def __len__(self):
            return len(self.dataset)
    transformed_dataset=TransformedDataset(dataset, transform=trans)
    dataloader=DataLoader(transformed_dataset,batch_size=batch_size)
    return dataloader

#创建损失计算函数
def evaluate_loss(net,data_iter,loss,device):
    net.eval()
    total_loss=0
    total_len=0
    with torch.no_grad():
        for x,y in data_iter:
            x=x.to(device)
            y=y.to(device)
            l=loss(net(x),y)
            total_loss+=l*len(x)
            total_len+=len(x)
        total_loss=total_loss.to('cpu')
    net.train()
    return total_loss/total_len

#创建准确率计算函数
def accumulate_num(y_true,y_pred):
    return (y_true==y_pred).sum().item()
def evaluate_accuracy(net,data_iter,device):
    net.eval()
    accurate_num=0
    total_len=0
    with torch.no_grad():
        for x,y in data_iter:
            x=x.to(device)
            y=y.to(device)
            y_pred=net(x).argmax(dim=1)
            accurate_num+=accumulate_num(y,y_pred)
            total_len+=len(x)
    net.train()
    return accurate_num/total_len

#创建训练函数
def train(net,train_iter,loss,trainer,epochs,device):
    net.train()
    epochs=[i for i in range(1,epochs+1)]
    loss_count=[]
    accuracy_count=[]
    for epoch in epochs:
        for x,y in train_iter:
            x=x.to(device)
            y=y.to(device)
            l=loss(net(x),y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        loss_count.append(evaluate_loss(net,train_iter,loss,device))
        accuracy_count.append(evaluate_accuracy(net,train_iter,device))
        print(f"epoch:{epoch},loss:{loss_count[-1]}，accuracy:{accuracy_count[-1]}")
    #可视化
    plt.plot(epochs,loss_count,label='loss')
    plt.plot(epochs,accuracy_count,label='accuracy')
    plt.legend()
    plt.show()

#创建预测函数并保存为CSV文件
def predict(net,test_iter,save_path,device):
    net.eval()
    y_pred=[]
    with torch.no_grad():
        for x in test_iter:
            x=x.to(device)
            batch_result=net(x).argmax(dim=1).to('cpu')
            batch_result=batch_result.tolist()
            y_pred.extend(batch_result)
    results=pd.DataFrame({'label':y_pred})
    results.to_csv(save_path,index=False)
    net.train()

#读取预测的CSV文件将其逆编码成标签
def inverse_label(train_csv,results_csv):
    train=pd.read_csv(train_csv)
    results=pd.read_csv(results_csv)
    #让LabelEncoder拟合训练集
    encoder=LabelEncoder()
    encoder.fit(train['label'])
    results['label']=encoder.inverse_transform(results['label'])
    results.to_csv(results_csv,index=False)

#对模型预测结果进行加权平均
def voting(weights,results,categories):
    '''
    :param weights: 权重列表
    :param results: 存储结果的列表
    categories:类别数目
    :return: 加权平均后的结果
    '''
    samples=len(results[0]) #样本个数
    ans=np.array([[0 for i in range(0,categories)]for j in range(0,samples)])
    #将结果标签独热编码
    trans=LabelBinarizer()
    for i in range(0,len(results)):
        if i==0:
            results[i]=trans.fit_transform(results[i])
        else:
            results[i]=trans.transform(results[i])
    #计算加权平均
    for i in range(0,samples):
        score=np.array([0.0 for z in range(0,categories)])
        for j in range(0,len(results)):
            score+=results[j][i]*weights[j]
        label_position=score.argmax()
        ans[i][label_position]=1
    ans=trans.inverse_transform(ans)
    return ans



