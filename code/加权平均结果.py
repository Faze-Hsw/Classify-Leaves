import pandas as pd
from base import voting
#读取三个模型的结果并stack
densenet_121=pd.read_csv(r"C:\Users\TS.1989\Desktop\黄枢炜\项目\Classify Leaves\data\results_densenet121.csv")['label'].values.tolist()
resnet_18=pd.read_csv(r"C:\Users\TS.1989\Desktop\黄枢炜\项目\Classify Leaves\data\results_resnet18.csv")['label'].values.tolist()
resnet_50=pd.read_csv(r"C:\Users\TS.1989\Desktop\黄枢炜\项目\Classify Leaves\data\results_resnet50.csv")['label'].values.tolist()
results=[]
results.append(densenet_121)
results.append(resnet_18)
results.append(resnet_50)
#进行加权平均
ans=voting([1.0,1.0,1.5],results,categories=176)
#保存结果
ans=pd.DataFrame({'label':ans})
ans.to_csv(r"C:\Users\TS.1989\Desktop\黄枢炜\项目\Classify Leaves\data\results.csv",index=False)
