import pandas as pd
import numpy as np
import seaborn as sns

import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random

import heapq

processedDataStorePath = '../processed_data/'


def testXGBoost(X_train,y_train,X_test,y_test):
    
    dtrain = xgb.DMatrix(data = X_train, label = y_train)
#     Dval = xgb.DMatrix(data=X_val, label=y_val)
    Dtest = xgb.DMatrix(data = X_test, label = y_test)
    
    param = {'max_depth': 3, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic','estimators':10,'eval_metric':'auc'}

#     evallist = [(Dval, 'eval')]
    evallist = [(dtrain, 'train')]
    plst = param.items()

    num_round = 10

    bst = xgb.train(plst, dtrain, num_round, evallist, early_stopping_rounds=5)#, verbose_eval=50)
    y_pred_score = bst.predict(Dtest)
    y_pred = [round(value) for value in y_pred_score]

    print(confusion_matrix(y_test,y_pred),'\n')
    print(accuracy_score(y_test,y_pred),'\n')
    
    return bst,accuracy_score(y_test,y_pred)


def getIndexWithHighConfidence(data,p):
    # print('=========================================')
    # print('=========================================')
    largestList=[]
    smallestList=[]
    largestValue = heapq.nlargest(1,data)[0]
    print('largest value: ', largestValue)
    smallestValue = heapq.nsmallest(1,data)[0]
    print('smallest value: ', smallestValue)
    if largestValue>.7:
        largestList = [idx for idx, val in enumerate(data) if abs(val-largestValue)<=p]
#         if (len(largestList)>num):
#             random.shuffle(largestList)
#             largestList=largestList[:num]
    print('len of largest list: ', len(largestList))
    if smallestValue<.3:
        smallestList = [idx for idx, val in enumerate(data) if abs(val-smallestValue)<=p]
#         if (len(smallestList)>num):
#             random.shuffle(smallestList)
#             smallestList=smallestList[:num]
    print('len of smallest list: ', len(smallestList))
    indexList = largestList + smallestList

    # print('=========================================')
    # print('=========================================')
    
    indexList.sort()
    indexList.reverse()
    return indexList

from matplotlib import pyplot as plt
def drawLine(data):
    dataRange = [i for i in range(1,len(data)+1)]
    df = pd.DataFrame({'x':dataRange, 'y':data})
    plt.plot( 'x', 'y', data=df, marker='o', color='mediumvioletred')
    plt.show()
    
    
def selfTraining(X_train_labeled,y_train_labeled,X_test,y_test,X_train_unlabeled):

    accuracy=[]
    prob = 0.002

    bst,acc = testXGBoost(X_train_labeled,y_train_labeled,X_test,y_test)
    accuracy.append(acc)

    try:
        for i in range(100):
            y_pseudo_score = bst.predict(xgb.DMatrix(data=X_train_unlabeled, label=[1 for i in range(len(X_train_unlabeled))]))
            y_pseudo = np.array(list(map(int,[np.round(i) for i in y_pseudo_score])))

            X_new=[]
            y_new=[]

            for index in getIndexWithHighConfidence(y_pseudo_score,prob):
                X_new.append(X_train_unlabeled[index])
                y_new.append(y_pseudo[index])
                X_train_unlabeled=np.delete(X_train_unlabeled,index,0) # 0 is row

            X_new=np.array(X_new)
            y_new=np.array(y_new)
            print('number of newly added: ',len(y_new))

            X_train_labeled = np.concatenate((X_train_labeled,X_new),axis=0)
            y_train_labeled = np.concatenate((y_train_labeled,y_new),axis=0)

            if i%10==0:
                prob=prob*1.5


            bst,acc = testXGBoost(X_train_labeled,y_train_labeled,X_test,y_test)
            accuracy.append(acc)
    
    except:
        pass
    
    finally:
        drawLine(accuracy)

def coTraining(X_trainA_labeled,X_trainB_labeled,y_train_labeled,X_testA,X_testB,y_test,X_trainA_unlabeled,X_trainB_unlabeled):
    
    fa = [1, 3, 5, 7, 9, 11, 12, 13] # feature A, refer to data_process.ipynb
    fb = [0, 2, 4, 6, 8, 10, 14, 15]
    
    accuracy=[]
    prob = 0.002

    bstA,accA = testXGBoost(X_trainA_labeled,y_train_labeled,X_testA,y_test)
    bstB,accB = testXGBoost(X_trainB_labeled,y_train_labeled,X_testB,y_test)
    
    accuracy.append((accA+accB)/2)

    try:
        for i in range(100):
            yA_pseudo_score = bstA.predict(xgb.DMatrix(data=X_trainA_unlabeled, label=[1 for i in range(len(X_trainA_unlabeled))]))
            yB_pseudo_score = bstB.predict(xgb.DMatrix(data=X_trainB_unlabeled, label=[1 for i in range(len(X_trainB_unlabeled))]))
            yA_pseudo = np.array(list(map(int,[np.round(i) for i in yA_pseudo_score])))

            highConfidenceIndexA = getIndexWithHighConfidence(yA_pseudo_score,prob)
            highConfidenceIndexB = getIndexWithHighConfidence(yB_pseudo_score,prob)
        
            highConfidenceIndexA.extend(highConfidenceIndexB)
            highConfidenceIndex = list(set(highConfidenceIndexA))
            
            # agree
            highConfidenceIndex = [index for index in highConfidenceIndex if np.round(yA_pseudo_score[index])==np.round(yB_pseudo_score[index])]
            highConfidenceIndex.sort()
            highConfidenceIndex.reverse()
            
            X_newA=[]
            X_newB=[]
            y_new=[]
            
            for index in highConfidenceIndex: 
                X_newA.append(X_trainA_unlabeled[index])
                X_newB.append(X_trainB_unlabeled[index])
                y_new.append(yA_pseudo[index])
                X_trainA_unlabeled=np.delete(X_trainA_unlabeled,index,0) # 0 is row
                X_trainB_unlabeled=np.delete(X_trainB_unlabeled,index,0)
                
            X_newA=np.array(X_newA)
            X_newB=np.array(X_newB)
            y_new=np.array(y_new)
            
            print('number of newly added: ',len(y_new))

            X_trainA_labeled = np.concatenate((X_trainA_labeled,X_newA),axis=0)
            X_trainB_labeled = np.concatenate((X_trainB_labeled,X_newB),axis=0)
            y_train_labeled = np.concatenate((y_train_labeled,y_new),axis=0)

            if i%10==0:
                prob=prob*1.5


            bstA,accA = testXGBoost(X_trainA_labeled,y_train_labeled,X_testA,y_test)
            bstB,accB = testXGBoost(X_trainB_labeled,y_train_labeled,X_testB,y_test)

            accuracy.append((accA+accB)/2)
            
    except:
        pass
    
    finally:
        drawLine(accuracy)


def main():

	for i in range(1,6):

		print('\n\n=========================================')
		print('=========================================')
		print('training and testing set ', i)
		print('=========================================')
		print('=========================================\n\n')

		X_train_labeled = np.load(processedDataStorePath+str(i)+'X_train_labeled.npy')
		y_train_labeled = np.load(processedDataStorePath+str(i)+'y_train_labeled.npy')
		X_trainA_labeled = np.load(processedDataStorePath+str(i)+'X_trainA_labeled.npy')
		# y_trainA_labeled = np.load(processedDataStorePath+'y_trainA_labeled.npy')
		X_trainB_labeled = np.load(processedDataStorePath+str(i)+'X_trainB_labeled.npy')
		# y_trainB_labeled = np.load(processedDataStorePath+'y_trainB_labeled.npy')
		X_train_unlabeled = np.load(processedDataStorePath+str(i)+'X_train_unlabeled.npy')
		X_test = np.load(processedDataStorePath+str(i)+'X_test.npy')
		y_test = np.load(processedDataStorePath+str(i)+'y_test.npy')

		fa = [1, 3, 5, 7, 9, 11, 12, 13] # feature A, refer to data_process.ipynb
		fb = [0, 2, 4, 6, 8, 10, 14, 15]

		def splitFeatures(dataSet, featureList):
		    return (dataSet.T[featureList]).T

		X_testA = splitFeatures(X_test, fa)
		X_testB = splitFeatures(X_test, fb)
		X_trainA_unlabeled = splitFeatures(X_train_unlabeled, fa)
		X_trainB_unlabeled = splitFeatures(X_train_unlabeled, fb)

		part=10
		start=5


		bst,acc = testXGBoost(X_train_labeled[start:part+start],y_train_labeled[start:part+start],X_test,y_test)
		print('=========================================')
		print('all features: ', acc)
		print('=========================================\n\n')

		print('=========================================')
		print('all features self training: ')
		print('=========================================')
		selfTraining(X_train_labeled[start:part+start],y_train_labeled[start:part+start],X_test,y_test,X_train_unlabeled)
		


		bstA,accA = testXGBoost(X_trainA_labeled[start:part+start],y_train_labeled[start:part+start],X_testA,y_test)
		bstB,accB = testXGBoost(X_trainB_labeled[start:part+start],y_train_labeled[start:part+start],X_testB,y_test)
		print('=========================================')
		print('cotraining: : ', accA, accB)
		print('=========================================\n\n')
		print('=========================================')
		print('cotraining:')
		print('=========================================')
		coTraining(X_trainA_labeled[start:part+start],X_trainB_labeled[start:part+start],y_train_labeled[start:part+start],X_testA,X_testB,y_test,X_trainA_unlabeled,X_trainB_unlabeled)


if __name__=='__main__':
	main()

