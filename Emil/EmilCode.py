import time
import math
import operator
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import make_scorer, log_loss
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from future_encoder import ColumnTransformer
from sklearn.utils import shuffle, resample
from sklearn.base import clone
from os import chdir, getcwd
# from google.colab import drive

PATH = 'originalData/'

def main():
  if getcwd() != '/Users/heyuhao/Desktop/cz4041_semisupervised/Emil':
    # drive.mount('/content/gdrive')
    chdir('/Users/heyuhao/Desktop/cz4041_semisupervised/Emil')

  DATASETS_10 = ['abalone-ssl10', 'australian-ssl10', 'banana-ssl10', 'coil2000-ssl10', 'magic-ssl10', 'spambase-ssl10', 'splice-ssl10']
  DATASETS_40 = ['abalone-ssl40', 'australian-ssl40', 'banana-ssl40', 'coil2000-ssl40', 'magic-ssl40', 'spambase-ssl40', 'splice-ssl40']
  ALL_DATASETS = DATASETS_10 + DATASETS_40
  CV_folds = 10

  CLASSIFIER_NAMES = [
      'Decision Tree',
      'k-Nearest Neighbors',
      'Support-vector Machine'
  ]

  CLASSIFIERS = [
      DecisionTreeClassifier(max_depth = 5),
      KNeighborsClassifier(),
      SVC(gamma='auto', tol=0.005, probability=True)
  ]

  WRAPPER_NAMES = [
      'Self-training',
      'Tri-training',
      'Tri-training with disagreement'
  ]
  WRAPPERS = [
      self_training,
      tri_training,
      tri_training_with_disagreement
  ]
  
  MULTI_NAMES = [
      'Democratic Co-learning'
  ]
  MULTI = [
      democratic_co_learning
  ]
  
  DROP_RATES = [
      0.0,
      0.5
  ]
  
  print('Running experiments with {} data sets, {} base classifiers, {} wrappers, {} multi-classifier methods and {} dropout rates.'.format(
      len(ALL_DATASETS), len(CLASSIFIER_NAMES), len(WRAPPER_NAMES), len(MULTI_NAMES), len(DROP_RATES)))
  for dataset in ALL_DATASETS:
    print('\n-----> Data set: {}'.format(dataset))
    X_tra_folds = []
    U_tra_folds = []
    Y_tra_folds = []
    X_tst_folds = []
    Y_tst_folds = []
    
    # Read in data
    for i in range(CV_folds):
      X_tra, U_tra, Y_tra , X_tst, Y_tst = prepare_data(dataset, i+1)
      X_tra_folds.append(X_tra)
      U_tra_folds.append(U_tra)
      Y_tra_folds.append(Y_tra)
      X_tst_folds.append(X_tst)
      Y_tst_folds.append(Y_tst)
      
    # Determine baseline accuracies for each classifier
    for i, clf in enumerate(CLASSIFIERS):
      acc = []
      for j in range(CV_folds):
        clf.fit(X_tra_folds[j], Y_tra_folds[j])
        acc.append(clf.score(X_tst_folds[j], Y_tst_folds[j]))
      
      cv_acc = sum(acc) / CV_folds
      print('{} baseline accuracy: {:.3f}'.format(CLASSIFIER_NAMES[i], cv_acc))
    
    
    for i, wrapper in enumerate(WRAPPERS):
      for drop_rate in DROP_RATES:
        if drop_rate == 0:
          print('\n---> Algorithm: {} w/o dropouts'.format(WRAPPER_NAMES[i]))
        else:
          print('\n---> Algorithm: {} w/ dropout rate {}'.format(WRAPPER_NAMES[i], drop_rate))
          
        for j, clf in enumerate(CLASSIFIERS):
          print('-> Classifier: {}'.format(CLASSIFIER_NAMES[j]))

          acc = []
          print('Cross-validation round', end=' ')
          for k in range(CV_folds):
            print(k , end=' ')
            acc.append(wrapper(clf, X_tra_folds[k], U_tra_folds[k], Y_tra_folds[k], X_tst_folds[k], Y_tst_folds[k], drop_rate))

          cv_acc = sum(acc) / CV_folds
          print('\nCross-validation accuracy: {:.3f}'.format(cv_acc))
        
    for i, multi in enumerate(MULTI):
      for drop_rate in DROP_RATES:
        if drop_rate == 0:
          print('\n---> Algorithm: {} w/o dropouts'.format(MULTI_NAMES[i]))
        else:
          print('\n---> Algorithm: {} w/ dropout rate {}'.format(MULTI_NAMES[i], drop_rate))

        acc = []
        print('Cross-validation round', end=' ')
        for k in range(CV_folds):
          print(k , end=' ')
          acc.append(multi(CLASSIFIERS, X_tra_folds[k], U_tra_folds[k], Y_tra_folds[k], X_tst_folds[k], Y_tst_folds[k], drop_rate))

        cv_acc = sum(acc) / CV_folds
        print('\nCross-validation accuracy: {:.3f}'.format(cv_acc))
      

def self_training(clf, X_train, U_train, Y_train, X_test, Y_test, drop_rate=0):
  U_residual = U_train
  U_augmentation = np.empty((0, U_train.shape[1]))
  U_labels = np.empty(0)
  
  threshold = 0.10
  max_iterations = 30
  
  for i in range(max_iterations):
    if U_residual.shape[0] == 0:
      break
      
    U_augmentation, U_labels = drop(U_augmentation, U_labels, drop_rate)
    X_augmented = np.concatenate([X_train, U_augmentation])
    Y_augmented = np.concatenate([Y_train, U_labels])
    
    clf.fit(X_augmented, Y_augmented)
    
    U_pred = clf.predict(U_train)
    U_prob = np.amax(clf.predict_proba(U_train), axis=1)
    
    U_res_prob = np.amax(clf.predict_proba(U_residual), axis=1)
    prob_threshold = np.amax(U_res_prob) - threshold
    
    aug_idx = np.nonzero(np.where(U_prob > prob_threshold, 1, 0))
    U_augmentation = U_train[aug_idx]
    U_labels = U_pred[aug_idx]
    
    res_idx = np.nonzero(np.where(U_prob > prob_threshold, 0, 1))
    U_residual = U_train[res_idx]
    
  return clf.score(X_test, Y_test)


def democratic_co_learning(clfs, X_train, U_train, Y_train, X_test, Y_test, drop_rate=0):
  num_of_clfs = len(clfs)
  U_idx = [[]] * num_of_clfs
  Y = [[]] * num_of_clfs
  e = [0] * num_of_clfs
  
  stopping_criteria = False
  while not stopping_criteria:
    w = []
    l = []
    for i in range(num_of_clfs):
      U_augmentation, U_labels = drop(U_train[U_idx[i]], np.array(Y[i]), drop_rate)
          
      X_augmented = np.concatenate((X_train, U_augmentation))
      Y_augmented = np.concatenate((Y_train, U_labels))
      clfs[i].fit(X_augmented, Y_augmented)
      acc = clfs[i].score(X_train, Y_train)
      acc_l = acc - 1.96 * math.sqrt(acc * (1 - acc) / Y_train.shape[0])
      w.append(acc)
      l.append(acc_l)
    
    X_prime = [[]] * num_of_clfs
    Y_prime = [[]] * num_of_clfs
    
    for x_idx, x in enumerate(U_train):
      c = []
      for clf in clfs:
        predicted_class = clf.predict(x.reshape(1, -1))
        c.append(predicted_class[0])
        
      ck = majority_class(c)
      if ck != -1:
        w_sum = {}
        for i in range(num_of_clfs):
          if c[i] in w_sum:
            w_sum[c[i]] += w[i]
          else:
            w_sum[c[i]] = w[i]
        if max(w_sum.items(), key=operator.itemgetter(1))[0] == ck:
          for i in range(num_of_clfs):
            if c[i] != ck and x_idx not in U_idx[i]:
              X_prime[i] = X_prime[i] + [x_idx]
              Y_prime[i] = Y_prime[i] + [ck]
           
    for i in range(num_of_clfs):
      if len(X_prime[i]) > 0:
        Li_size = len(Y[i]) + Y_train.shape[0]
        q = Li_size * (1 - 2*(e[i]/Li_size))**2

        Li_prime_size = len(Y_prime[i])
        acc_approx = sum(l[:i]) + sum(l[i+1:]) / (num_of_clfs - 1)
        e_prime = Li_prime_size * (1 - acc_approx)
        
        union_size = Li_size + Li_prime_size
        q_prime = union_size * (1 - 2*(e[i] + e_prime)/union_size)**2
        
        if q_prime > q:
          U_idx[i] = U_idx[i] + X_prime[i]
          Y[i] = Y[i] + Y_prime[i]
          e[i] += e_prime
        else:
          X_prime[i] = []
    
    stopping_criteria = all(len(x) == 0 for x in X_prime)
  
  test_preds = []
  for i in range(num_of_clfs):
    test_preds.append(clfs[i].predict(X_test))
  
  final_preds = majority_vote(test_preds)
  num_of_correct = np.nonzero(np.where(final_preds == Y_test, 1, 0))[0].shape[0]
  acc = num_of_correct / Y_test.shape[0]
  
  return acc
      
      
def majority_class(predictions):
  num_of_votes = {}
  
  for j in range(len(predictions)):
    vote = predictions[j]

    if vote in num_of_votes:
      num_of_votes[vote] += 1
    else:
      num_of_votes[vote] = 1

  winner = max(num_of_votes.items(), key=operator.itemgetter(1))[0]
  if num_of_votes[winner] > len(predictions) / 2:
    return winner
  else:
    return -1

  
def row_in_array(row, array): 
  return any(np.array_equal(row, x) for x in array)


def tri_training(clf, X_train, U_train, Y_train, X_test, Y_test, drop_rate=0):
  num_of_clfs = 3
  clfs = []
  e_prime = []
  l_prime = []
  
  # Perform bootstrapping
  for i in range(num_of_clfs):
    sampled_X, sampled_Y = resample(X_train, Y_train)
    clf_clone = clone(clf)
    clfs.append(clf_clone.fit(sampled_X, sampled_Y))
    e_prime.append(1 - 1/np.unique(Y_train).shape[0])
    l_prime.append(0)

  stopping_criteria = False
  while not stopping_criteria:

    update = [False, False, False]
    X_augmented = [False, False, False]
    Y_augmented = [False, False, False]
    e = [False, False, False]
    l = [False, False, False]
    
    for i in range(num_of_clfs):
      e[i] = measure_error(clfs[i-2], clfs[i-1], X_train, Y_train)
      
      if e[i] < e_prime[i]:
        pred1 = clfs[i-1].predict(U_train)
        pred2 = clfs[i-2].predict(U_train)
        agreement_idx = np.nonzero(np.where(pred1 == pred2, 1, 0))
        l[i] = agreement_idx[0].shape[0]
        
        if l_prime[i] == 0:
          l_prime[i] = math.floor(e[i]/(e_prime[i] - e[i]) + 1)
        
        if l_prime[i] < l[i]:
          if e[i]*l[i] < e_prime[i]*l_prime[i]:
            update[i] = True
          elif l_prime[i] > e[i]/(e_prime[i] - e[i]):
            agreement_idx = (resample(agreement_idx[0], replace=False, n_samples=math.ceil(e_prime[i]*l_prime[i]/e[i] - 1)))
            update[i] = True
            
        if update[i]:
          U_augmentation, U_labels = drop(U_train[agreement_idx], pred1[agreement_idx], drop_rate)
          
          X_augmented[i] = np.concatenate((X_train, U_augmentation))
          Y_augmented[i] = np.concatenate((Y_train, U_labels))
          
    
    for i in range(num_of_clfs):
      if update[i]:
        clfs[i] = clfs[i].fit(X_augmented[i], Y_augmented[i])
        e_prime[i] = e[i]
        l_prime[i] = l[i]
        
    stopping_criteria = all(not up for up in update)
  
  test_preds = []
  for i in range(num_of_clfs):
    test_preds.append(clfs[i].predict(X_test))
  
  final_preds = majority_vote(test_preds)
  num_of_correct = np.nonzero(np.where(final_preds == Y_test, 1, 0))[0].shape[0]
  acc = num_of_correct / Y_test.shape[0]
  
  return acc


def tri_training_with_disagreement(clf, X_train, U_train, Y_train, X_test, Y_test, drop_rate=0):
  num_of_clfs = 3
  clfs = []
  e_prime = []
  l_prime = []
  X_sets = []
  Y_sets = []
  
  # Perform bootstrapping
  for i in range(num_of_clfs):
    sampled_X, sampled_Y = resample(X_train, Y_train)
    X_sets.append(sampled_X)
    Y_sets.append(sampled_Y)
    clf_clone = clone(clf)
    clfs.append(clf_clone.fit(sampled_X, sampled_Y))
    e_prime.append(1 - 1/np.unique(Y_train).shape[0])
    l_prime.append(0)

  stopping_criteria = False
  while not stopping_criteria:

    update = [False, False, False]
    X_augmented = [False, False, False]
    Y_augmented = [False, False, False]
    e = [False, False, False]
    l = [False, False, False]
    
    for i in range(num_of_clfs):
      e[i] = measure_error(clfs[i-2], clfs[i-1], X_train, Y_train)
      
      if e[i] < e_prime[i]:
        pred0 = clfs[i].predict(U_train)
        pred1 = clfs[i-1].predict(U_train)
        pred2 = clfs[i-2].predict(U_train)
        agreement_idx = np.nonzero(np.where(pred1 == pred2, 1, 0))
        disagreement_idx = np.nonzero(np.where(pred1 != pred0, 1, 0))
        augmentation_idx = (np.intersect1d(agreement_idx, disagreement_idx), )
        
        l[i] = augmentation_idx[0].shape[0]
        
        if l_prime[i] == 0:
          l_prime[i] = math.floor(e[i]/(e_prime[i] - e[i]) + 1)
        
        if l_prime[i] < l[i]:
          if e[i]*l[i] < e_prime[i]*l_prime[i]:
            update[i] = True
          elif l_prime[i] > e[i]/(e_prime[i] - e[i]):
            augmentation_idx = (resample(augmentation_idx[0], replace=False, n_samples=math.ceil(e_prime[i]*l_prime[i]/e[i] - 1)))
            update[i] = True
            
        if update[i]:
          U_augmentation, U_labels = drop(U_train[augmentation_idx], pred1[augmentation_idx], drop_rate)
          
          X_augmented[i] = np.concatenate((X_train, U_augmentation))
          Y_augmented[i] = np.concatenate((Y_train, U_labels))
          
    
    for i in range(num_of_clfs):
      if update[i]:
        clfs[i] = clfs[i].fit(X_augmented[i], Y_augmented[i])
        e_prime[i] = e[i]
        l_prime[i] = l[i]
        
    stopping_criteria = all(not up for up in update)
  
  test_preds = []
  for i in range(num_of_clfs):
    test_preds.append(clfs[i].predict(X_test))
  
  final_preds = majority_vote(test_preds)
  num_of_correct = np.nonzero(np.where(final_preds == Y_test, 1, 0))[0].shape[0]
  acc = num_of_correct / Y_test.shape[0]
  
  return acc
  
  
def measure_error(clf1, clf2, X, Y):
  pred1 = clf1.predict(X)
  pred2 = clf2.predict(X)
  agreement_idx = (np.nonzero(np.where(pred1 == pred2, 1, 0)),)
  
  preds = pred1[agreement_idx]
  truth = Y[agreement_idx]
  
  num_of_incorrect = np.nonzero(np.where(preds != truth, 1, 0))[0].shape[0]
  err = num_of_incorrect / truth.shape[0]
  
  return err
  
  
def majority_vote(predictions):
  num_of_predictions = predictions[0].shape[0]
  
  res = np.empty_like(predictions[0])
  for i in range(num_of_predictions):
    
    num_of_votes = {}
    for j in range(len(predictions)):
      vote = predictions[j][i]
      
      if vote in num_of_votes:
        num_of_votes[vote] += 1
      else:
        num_of_votes[vote] = 1
        
    res[i] = max(num_of_votes.items(), key=operator.itemgetter(1))[0]
  
  return res
  
  
def drop(X, Y, drop_rate):
  keep_idx = np.nonzero(np.where(np.random.random_sample(Y.shape) > drop_rate, 1, 0))
  
  return X[keep_idx], Y[keep_idx]


def prepare_data(dataset, i):
  """Reads data from cross-validation file i and standardizes inputs."""
  tra_path = PATH + dataset + '/' + dataset + '-10-' + str(i) + 'tra.dat'
  tst_path = PATH + dataset + '/' + dataset + '-10-' + str(i) + 'tst.dat'

  with open(tra_path, 'r') as f:
    tra_lines = f.read().splitlines()[1:] # Skip @relation

  attributes = read_attributes(tra_lines)
  X_tra, U_tra, Y_tra = read_data(tra_lines)

  with open(tst_path, 'r') as f:
    tst_lines = f.read().splitlines()[1:] # Skip @relation

  X_tst, _, Y_tst = read_data(tst_lines)
    
  X_tra, U_tra, X_tst = standardize_inputs(X_tra, U_tra, X_tst, attributes)
  Y_tra, Y_tst = standardize_outputs(Y_tra, Y_tst)
  
  return X_tra, U_tra, Y_tra , X_tst, Y_tst


def standardize_inputs(X_tra, U_tra, X_tst, attributes):
  cols = list(map(lambda x: x[0], attributes))
  X_tra_df = pd.DataFrame(X_tra, columns = cols)
  U_tra_df = pd.DataFrame(U_tra, columns = cols)
  X_tst_df = pd.DataFrame(X_tst, columns = cols)
  
  numerical_attributes, categorical_attributes = [], []
  for attr_name, attr_type in attributes:
    if attr_type == 'num':
      numerical_attributes.append(attr_name)
      X_tra_df[attr_name] = X_tra_df[attr_name].astype(float)
      U_tra_df[attr_name] = U_tra_df[attr_name].astype(float)
      X_tst_df[attr_name] = X_tst_df[attr_name].astype(float)
    elif attr_type == 'cat':
      categorical_attributes.append(attr_name)
    else:
      raise TypeError("Attribute type for attribute {} must be either 'cat' or 'num'. Found: '{}'".format(attr, attr_type))

  numerical_transformer = StandardScaler()
  categorical_transformer = OneHotEncoder( handle_unknown='ignore')
  
  standardizer = ColumnTransformer(sparse_threshold=0,
    transformers=[
        ('num', numerical_transformer, numerical_attributes),
        ('cat', categorical_transformer, categorical_attributes)])
  
  # Fit on both labeled and unlabeled training examples
  standardizer.fit(pd.concat((X_tra_df, U_tra_df)))
  
  X_tra_std = standardizer.transform(X_tra_df)
  U_tra_std = standardizer.transform(U_tra_df)
  X_tst_std = standardizer.transform(X_tst_df)
  
  return X_tra_std, U_tra_std, X_tst_std


def standardize_outputs(Y_tra, Y_tst):
  standardizer = LabelEncoder()
  
  # Fit on both train and test data to avoid having unknown categories in test
  standardizer.fit(Y_tra+Y_tst)
  Y_tra_std = standardizer.transform(Y_tra)
  Y_tst_std = standardizer.transform(Y_tst)
  
  return Y_tra_std, Y_tst_std


def read_data(lines):
  X, U, Y = [], [], []
  
  for line in lines:
    if line[0] == '@':
      continue

    values = line.split(', ')
    x = values[:-1]
    y = values[-1]

    if y == 'unlabeled':
      U.append(x)
    else:
      X.append(x)
      Y.append(y)
      
  return X, U, Y


def read_attributes(lines):
  attributes = []
  
  for line in lines:
    tag, value = line.split(' ', 1)
    
    if tag == '@attribute':
      attr_name, attr_type = value.split(' ', 1)
      if '{' == attr_type[0] and attr_type[-1] == '}':
        attributes.append((attr_name, 'cat'))
      else:
        attributes.append((attr_name, 'num'))
        
    else:
      return attributes[:-1] #Skip output
  
  
if __name__ == '__main__':
  main()