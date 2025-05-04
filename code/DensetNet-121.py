from base import train
from base import predict
from base import load_train_data,load_test_data
from base import inverse_label
from torchvision.models import densenet121,DenseNet121_Weights
from torch import nn
import torch
device=torch.device('cuda')
print(device)
#创建训练数据迭代器
train_iter=load_train_data(r"C:\Users\TS.1989\Desktop\黄枢炜\项目\Classify Leaves\data\train_images",r"C:\Users\TS.1989\Desktop\黄枢炜\项目\Classify Leaves\data\train.csv",batch_size=64)
#加载预训练的DenseNet-121模型
net=densenet121(weights=DenseNet121_Weights.DEFAULT)
#获取原始分类器的输入特征数量
in_features = net.classifier.in_features
#修改输出层以适配目标任务
net.classifier = torch.nn.Linear(in_features,176)
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
test_iter=load_test_data(r"C:\Users\TS.1989\Desktop\黄枢炜\项目\Classify Leaves\data\test_images",batch_size=64)
#输出结果
predict(net,test_iter,r"C:\Users\TS.1989\Desktop\黄枢炜\项目\Classify Leaves\data\results_densenet121.csv",device=device)
#逆编码输出结果
inverse_label(r"C:\Users\TS.1989\Desktop\黄枢炜\项目\Classify Leaves\data\train.csv",r"C:\Users\TS.1989\Desktop\黄枢炜\项目\Classify Leaves\data\results_densenet121.csv")