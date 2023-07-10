# sklearn
sklearn record  


## 0705 record
train01 使用2022-train-v2.csv  

測試不同模型 的rmse  
LinearRegression 9.6964  
Lasso  8.7243 但max_iter=1000無法收斂  
LassoLars 8.7513  
Ridge  9.3730  
BayesianRidge 8.8234  

## 0709 繼續測試
SVM(SVR) 13.2171  
Decision Tree 10.0090  
Random Forest 10.5705  但
目前測起來好像都無法把rmse壓下來  
現在只有把y設為第一欄，但實際上會有六欄，這樣只有測一欄好像不準，全部測再平均?  
sklearn官網上的decision tree 中有提到 Multi-output 和 missing value support 之後研究  

