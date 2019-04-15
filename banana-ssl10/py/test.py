
# coding: utf-8

# In[1]:


#some techniques may need

# 1. concatenate 2 np array
# a=np.array([[1,2]])
# b=np.array([[1,2]])
# print(np.concatenate((a, b), axis=0)) -- print[[1,2],[1,2]]

# 2. table column unique value counts
# print(y_labeled.value_counts())

# 3. np array unique value counts
# unique, counts = np.unique(y_labeled, return_counts=True)
# print(np.asarray((unique, counts)).T) 


# ## --------------------------
# ## Get data
# ## --------------------------

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns

processedDataStorePath = '../processed_data/'


# In[3]:


X_labeled = np.load(processedDataStorePath+'X_labeled.npy')
y_labeled = np.load(processedDataStorePath+'y_labeled.npy')

X_unlabeled = np.load(processedDataStorePath+'X_unlabeled.npy')
y_unlabeled = np.load(processedDataStorePath+'y_unlabeled.npy')


# In[7]:


unique, counts = np.unique(y_labeled, return_counts=True)
print('for labeled data: \n', np.asarray((unique, counts)).T) 

unique, counts = np.unique(y_unlabeled, return_counts=True)
print('for unlabeled data: \n', np.asarray((unique, counts)).T) 


# In[8]:


X_labeled[:5]


# In[11]:


def drawDf(x1,x2,label,df):
    sns.lmplot(x1, x2, data=df, fit_reg=False,  
        scatter_kws={"s": 100}, # marker size
        palette='Set1',
        hue=label)
    
def drawNpArray(x1,x2,y):
    df = pd.DataFrame({'f1':x1,'f2':x2,'label':y})
    drawDf('f1','f2','label',df)
    
print('\nall labeled data distribution graph: ')
drawNpArray(X_labeled.T[0],X_labeled.T[1],y_labeled)


# ## --------------------------
# ## Testing
# ## --------------------------

# In[28]:


import random

from sklearn.model_selection import train_test_split

randomList = [random.randint(1,100) for i in range(2)]
# randomList=[43,95]
X_trainVal, X_test, y_trainVal, y_test = train_test_split(X_labeled,y_labeled, test_size=0, random_state=randomList[0])
X_train, X_val, y_train, y_val = train_test_split(X_trainVal, y_trainVal, test_size=0.5, random_state=randomList[1])

print('random list is: ', randomList)

print('train: ', len(y_train), ' val: ',len(y_val), ' test: ', len(y_test))

drawNpArray(X_train.T[0],X_train.T[1],y_train)
drawNpArray(X_val.T[0],X_val.T[1],y_val)
# drawNpArray(X_test.T[0],X_test.T[1],y_test)


# In[19]:


import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random

def testXGBoost():

    dtrain = xgb.DMatrix(data = X_train, label = y_train)#, weight=weights)
    Dtest = xgb.DMatrix(data = X_test, label = y_test)
    Dval = xgb.DMatrix(data=X_val, label=y_val)
    
    drawNpArray(X_train.T[0],X_train.T[1],y_train)

    param = {'max_depth': 3, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic','estimators':500,'eval_metric':'auc'}

    evallist = [(Dval, 'eval')]
    plst = param.items()

    num_round = 500

    bst = xgb.train(plst, dtrain, num_round, evallist, early_stopping_rounds=10)
    y_pred_score = bst.predict(Dtest)
    y_pred = [round(value) for value in y_pred_score]

    print(confusion_matrix(y_test,y_pred),'\n')
    print(accuracy_score(y_test,y_pred),'\n')
    
    return bst


# In[20]:


bst=testXGBoost()


# In[21]:

# return index in reserve order
import heapq
def returnIndexOfPassingProb(data):

    largestValues = heapq.nlargest(1, data)
    print('largest value: ', largestValues)
    smallestValus = heapq.nsmallest(1, data)
    print('smallest value: ', smallestValus)

    largestList = [idx for idx, val in enumerate(data) if val in largestValues]
    if (len(largestList)>10):
        random.shuffle(largestList)
        largestList=largestList[:10]
    print('largest list: ', largestList)

    smallestList = [idx for idx, val in enumerate(data) if val in smallestValus]
    if (len(smallestList)>10):
        random.shuffle(smallestList)
        smallestList=smallestList[:10]
    print('smallest list: ', smallestList)

    indexList = largestList + smallestList

    indexList.sort()
    indexList.reverse()

    print(indexList)

    return indexList

# In[22]:

y_pseudo_score = bst.predict(xgb.DMatrix(data=X_unlabeled, label=[1 for i in y_unlabeled]))
returnList=returnIndexOfPassingProb(y_pseudo_score)

# In[34]:

bst=testXGBoost()

for i in range(10):
    y_pseudo_score = bst.predict(xgb.DMatrix(data=X_unlabeled, label=[1 for i in y_unlabeled]))

    X_new=[]
    y_new=[]

    for index in returnIndexOfPassingProb(y_pseudo_score):
        X_new.append(X_unlabeled[index])
        y_new.append(y_pseudo[index])
        X_unlabeled=np.delete(X_unlabeled,index,0) # 0 is row
        y_unlabeled=np.delete(y_unlabeled,index)

    X_new=np.array(X_new)
    y_new=np.array(y_new)
    print('number of newly added: ',len(y_new))

    X_train = np.concatenate((X_train,X_new),axis=0)
    y_train = np.concatenate((y_train,y_new),axis=0)

    bst = testXGBoost()

# In[23]:


# In[24]:


y_pseudo_score


# In[37]:






# In[24]:


# list(map(int,np.round([.6,.2])))


# In[25]:




