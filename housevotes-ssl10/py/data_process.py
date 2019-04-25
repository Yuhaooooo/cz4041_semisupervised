import pandas as pd
import numpy as np
import seaborn as sns

dataFramePath='../data_frame/'
processedDataStorePath = '../processed_data/'

def getDfArray(df, splitSign):
    totalDataList = []
    startingRow = 20
    endingRow = len(df)-1
    for i in range(startingRow, endingRow):
        dataString = df.iloc[i,0]
        dataList = dataString.split(splitSign)
        newList=[]
        for j in dataList:
            if j=='y' or j=='republican':
                newList.append(1)
            elif j=='n' or j=='democrat':
                newList.append(0)
            else:
                newList.append(-1)
        totalDataList.append(newList)
    return totalDataList

def main():
	for i in range(1,6):
		trainCsv = str(i)+'tra.csv'
		testCsv = str(i)+'tst.csv'
		df_train = pd.read_csv(dataFramePath+trainCsv)
		df_test = pd.read_csv(dataFramePath+testCsv)

		column = [df_train.loc[i]['@relation housevotes'][11:-5] for i in range(16)] + ['class'] #16 features and 1 label
		
		df_train = pd.DataFrame(np.array(getDfArray(df_train,', ')),columns=[j for j in range(17)])    
		df_test = pd.DataFrame(np.array(getDfArray(df_test,',')),columns=[j for j in range(17)])  

		df_train_labeled = df_train[df_train[16]!=-1]
		df_train_unlabeled = df_train[df_train[16]==-1]

		corrA = [1, 3, 5, 7, 9, 11, 12, 13]
		corrB = [0, 2, 4, 6, 8, 10, 14, 15]

		X_train_labeled = np.array(df_train_labeled.iloc[:,:-1])
		y_train_labeled = np.array(df_train_labeled.iloc[:,-1])

		df_trainA_labeled = df_train_labeled[corrA+[16]]
		X_trainA_labeled = np.array(df_trainA_labeled.iloc[:,:-1])

		df_trainB_labeled = df_train_labeled[corrB+[16]]  
		X_trainB_labeled = np.array(df_trainB_labeled.iloc[:,:-1])

		X_train_unlabeled = np.array(df_train_unlabeled.iloc[:,:-1])

		X_test = np.array(df_test.iloc[:,:-1])
		y_test = np.array(df_test.iloc[:,-1])

		np.save(processedDataStorePath+str(i)+'X_train_labeled.npy', X_train_labeled)
		np.save(processedDataStorePath+str(i)+'y_train_labeled.npy', y_train_labeled)
		np.save(processedDataStorePath+str(i)+'X_trainA_labeled.npy', X_trainA_labeled)
		np.save(processedDataStorePath+str(i)+'X_trainB_labeled.npy', X_trainB_labeled)
		np.save(processedDataStorePath+str(i)+'X_train_unlabeled.npy', X_train_unlabeled)
		np.save(processedDataStorePath+str(i)+'X_test.npy', X_test)
		np.save(processedDataStorePath+str(i)+'y_test.npy', y_test)

if __name__=='__main__':
	main()














