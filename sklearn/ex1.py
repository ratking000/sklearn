from sklearn.datasets import load_iris
data = load_iris()
x = data.data
y = data.target


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.1,random_state=0)

# svm分类器
from sklearn.svm import SVC
svm_model = SVC()
svm_model.fit(train_x,train_y)

print("params :", SVC().get_params())

quit()
# 一般驗證方法
from sklearn.metrics import accuracy_score
pred1 = svm_model.predict(train_x)
accuracy1 = accuracy_score(train_y,pred1)
print('在训练集上的精确度: %.4f'%accuracy1)
pred2 = svm_model.predict(test_x)
accuracy2 = accuracy_score(test_y,pred2)
print('在测试集上的精确度: %.4f'%accuracy2)

# 交叉驗證方法
# from sklearn.model_selection import cross_val_score
# scores1 = cross_val_score(svm_model,train_x,train_y,cv=5, scoring='accuracy')
# # 输出精确度的平均值和置信度区间
# print("训练集上的精确度: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))
# scores2 = cross_val_score(svm_model,test_x,test_y,cv=5, scoring='accuracy')
# # 输出精确度的平均值和置信度区间
# print("测试集上的平均精确度: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))
# print(scores1)
# print(scores2)