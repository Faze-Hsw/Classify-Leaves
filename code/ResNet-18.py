from base import train
from base import predict
from base import load_train_data,load_test_data
from base import inverse_label
from torchvision.models import resnet18,ResNet18_Weights
from torch import nn
import torch
device=torch.device('cuda')
print(device)
#创建训练数据迭代器
train_iter=load_train_data(r"C:\Users\TS.1989\Desktop\黄枢炜\项目\Classify Leaves\data\train_images",r"C:\Users\TS.1989\Desktop\黄枢炜\项目\Classify Leaves\data\train.csv",batch_size=256)
#加载预训练的ResNet-18模型
net=resnet18(weights=ResNet18_Weights.DEFAULT)
#修改模型的全连接层
num_ftrs=net.fc.in_features
net.fc=nn.Linear(num_ftrs,176)
nn.init.xavier_normal_(net.fc.weight)
net=net.to(device)
#创建损失函数
loss=nn.CrossEntropyLoss()
#创建优化器
trainer=torch.optim.Adam(net.parameters(),lr=1e-4)
#训练网络
train(net,train_iter,loss,trainer,epochs=20,device=device)
#删除训练数据迭代器以释放内存
del train_iter
#读取测试集数据
test_iter=load_test_data(r"C:\Users\TS.1989\Desktop\黄枢炜\项目\Classify Leaves\data\test_images",batch_size=256)
#输出结果
predict(net,test_iter,r"C:\Users\TS.1989\Desktop\黄枢炜\项目\Classify Leaves\data\results_resnet18.csv",device=device)
#逆编码输出结果
inverse_label(r"C:\Users\TS.1989\Desktop\黄枢炜\项目\Classify Leaves\data\train.csv",r"C:\Users\TS.1989\Desktop\黄枢炜\项目\Classify Leaves\data\results_resnet18.csv")