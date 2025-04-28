import pandas as pd
import os.path as osp
import numpy as np
from sklearn import multiclass
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from scipy.stats import pearsonr,spearmanr

work_path='results/IQA_fthnet_internal_ex4/visualization/eyeQS/'
result_csv_name='prediction_whole.csv'
fold=10
svm=False

test2=True if svm==False else False

raw=pd.read_csv(osp.join(work_path,result_csv_name))
df=pd.DataFrame(raw)

names=df['file_name'].tolist()
preds=df['prediction'].to_list()
gt=df['ground_truth'].to_list()

lens=int(len(names)/fold)
mat=[]

for ind in range(0,fold):
  sub_list=preds[0+ind*lens:(ind+1)*lens]
  mat.append(sub_list)

mat=np.array(mat)
preds2=mat.mean(axis=0)
names2=names[0:lens]
gt2=gt[0:lens]

recombine=pd.DataFrame({
  'file_name': names2,
  'prediction': preds2,
  'ground_truth': gt2,
})

save = recombine.to_csv(osp.join(work_path, 'prediction_recombine.csv'))

if svm==True:

  # 使用 lr 类，初始化模型
  lre = LogisticRegression(
      penalty="l2", C=1.0, random_state=None, solver="lbfgs", max_iter=3000,
      multi_class='ovr', verbose=0,
  )

  # 使用训练数据来学习（拟合），不需要返回值，训练的结果都在对象内部变量中
  preds2=np.reshape(preds2,(-1,1))
  gt2=(100*np.array(gt2)).astype(int).tolist()
  lre.fit(preds2, gt2)

  # 使用测试数据来预测，返回值预测分类数据
  y_pred = lre.predict(preds2)

  # 打印主要分类指标的文本报告
  print('--- report ---')
  print(classification_report(gt2, y_pred,digits=4))

  # 打印模型的参数
  print('--- params ---')
  print(lre.coef_, lre.intercept_)

  # 打印 auc
  # print('--- auc ---')
  # print(roc_auc_score(gt2, y_pred))

if test2==True:
  gt2=np.array(gt2)
  mse=np.sum((preds2-gt2)**2/len(preds2))
  rmse=np.sqrt(mse)
  srcc,_=spearmanr(preds2,gt2)
  plcc,_=pearsonr(preds2,gt2)

  print(f"MSE: {rmse}, SRCC: {srcc}, PLCC: {plcc}")


