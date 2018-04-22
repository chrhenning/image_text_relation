#!/usr/bin/env python3
# This script uses the extracted article embeddings to train a SVM

import argparse
import os
import json
import numpy as np
import pickle

from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")

# color codes (ANSI escape codes)
RED='\033[1;31m'
GREEN='\033[1;32m'
BLUE='\033[1;34m'
LIGHT_GREEN='\033[1;32m'
ORANGE='\033[1;33m'
NC='\033[0m' # No Color


def print_individual_metrics(y_true, y_pred, labels, print_labels, outobj, tag='', prependingSpaces=0):
	str_prec= 2

	heading = ['Label'] + print_labels
	row_format = ' ' * prependingSpaces + "# {:<15} | " + "{:>15}" * (len(heading) - 1)
	
	prec, rec, f1, supp = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None)
	round_arr = lambda arr : [round(x, str_prec) for x in arr]
	
	print(row_format.format(*heading))
	print(' ' * prependingSpaces + "# " + "-"*15  +"-|-" + "-" * 15 * (len(heading) - 1))
	print(row_format.format('Precision', *round_arr(prec)))
	print(row_format.format('Recall', *round_arr(rec)))
	print(row_format.format('F1-Score', *round_arr(f1)))
	print(row_format.format('Support', *supp))
	
	for i in range(len(labels)):
	  outobj[tag+'Precision of Label %d' % (print_labels[i])] = prec[i]
	  outobj[tag+'Recall of Label %d' % (print_labels[i])] = prec[i]
	  outobj[tag+'F1-Score of Label %d' % (print_labels[i])] = prec[i]
	  outobj[tag+'Support of Label %d' % (print_labels[i])] = prec[i]
	
	return outobj

def print_metrics(y_true, y_pred, labels, print_labels, tag='', outfile=None):
  outobj = {}

  acc = accuracy_score(y_true, y_pred)
  print('   # Overall Accuracy: ' + BLUE + str(acc) + NC)
  outobj[tag+'Accuracy'] = acc
  
  prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='micro')
  print('   #')
  print('   # Micro Precision: ' + str(prec))
  print('   # Micro Recall: ' + str(rec))
  print('   # Micro F1-Score: ' + LIGHT_GREEN + str(f1) + NC)
  outobj[tag+'Micro Precision'] = prec
  outobj[tag+'Micro Recall'] = rec
  outobj[tag+'Micro F1-Score'] = f1
  
  prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='macro')
  print('   #')
  print('   # Macro Precision: ' + str(prec))
  print('   # Macro Recall: ' + str(rec))
  print('   # Macro F1-Score: ' + LIGHT_GREEN + str(f1) + NC)
  outobj[tag+'Macro Precision'] = prec
  outobj[tag+'Macro Recall'] = rec
  outobj[tag+'Macro F1-Score'] = f1
  
  prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='weighted')
  print('   #')
  print('   # Weighted Precision: ' + str(prec))
  print('   # Weighted Recall: ' + str(rec))
  print('   # Weighted F1-Score: ' + LIGHT_GREEN + str(f1) + NC)
  outobj[tag+'Weighted Precision'] = prec
  outobj[tag+'Weighted Recall'] = rec
  outobj[tag+'Weighted F1-Score'] = f1
  
  print('   #')
  outobj = print_individual_metrics(y_true, y_pred, labels, print_labels, outobj, tag, 3)
  
  if outfile is not None:
    outstr = json.dumps(outobj)
    outfile.write(outstr + '\n')
  

def convertData(data):
  mi_labels = []
  sc_labels = []
  features = []
 
  for sample in data:
    mi_labels.append(sample['mi'])
    sc_labels.append(sample['sc'])
    features.append(sample['ae'])
    
  # convert sc_labels to int labels
  sc_labels_tmp = sc_labels
  sc_labels = []
  for sc in sc_labels_tmp:
    sc_labels.append(int(2*sc + 2))
      
  return mi_labels, sc_labels, features

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--intest', type=str, help="A json file containing the extracted test embeddings.", default='../classifier/extracted/test_samples.json')
  parser.add_argument('--intrain', type=str, help="A json file containing the extracted train embeddings.", default='../classifier/extracted/train_samples.json')
  #parser.add_argument('--C', type=float, help="The parameter C of sklearn.svm.LinearSVC.", default=1.0)
  parser.add_argument('--output', type=str, help="The computed metrics on the test set will be written into this file.", default='svm_results.jsonl')
  parser.add_argument('--simple', type=int, help="Set to 0, if simplified labeling is used (only 5 MI labels)", default=0)
  
  args = parser.parse_args()
  
  in_test = args.intest
  in_train = args.intrain
  #weight_decay = args.C
  outfile_fn = args.output
  simple_labeling = args.simple
  
  if not os.path.isfile(in_test):
    raise(Exception('The file ' + in_test + ' does not exists'))
    
  if not os.path.isfile(in_train):
    raise(Exception('The file ' + in_train + ' does not exists'))
  
  with open(in_test, 'r') as f:
    test_data = json.load(f)

  with open(in_train, 'r') as f:
    train_data = json.load(f)
    
  if os.path.isfile(outfile_fn):
    print('The file ' + outfile_fn + ' will be overwritten')
    
  outfile = open(outfile_fn, 'w')
   
  mi_labels_train, sc_labels_train, features_train =  convertData(train_data)
  mi_labels_test, sc_labels_test, features_test =  convertData(test_data)
  
  # Grid to search to find best weight decay parameter
  par_grid = [{'C': list(np.linspace(0.001,2, 1000))}]
  
  ### Mutual Information
  print('Train Multiclass SVM for Mutual Information ...')
  #lin_clf_mi = LinearSVC(C=weight_decay, multi_class='crammer_singer')
  lin_clf_mi = LinearSVC(multi_class='crammer_singer')
  gdCV = GridSearchCV(lin_clf_mi, par_grid, n_jobs=4)
  gdCV.fit(features_train, mi_labels_train) 
  print('Best parameter found via grid search: ' + str(gdCV.best_params_))
  # store the best MI classifier for later usage
  with open('best_mi_svm.pickle', 'wb') as f:
    pickle.dump(gdCV, f)
  
  mi_predict_train = gdCV.predict(features_train)
  mi_predict_test = gdCV.predict(features_test)
  
  if simple_labeling:
    mi_labels = list(range(5))
  else:
    mi_labels = list(range(8))
  
  print()
  print('   ### ' + RED + 'Mutual Information' + NC + ': Evaluation on ' + GREEN + 'Training Set' + NC)
  print_metrics(mi_labels_train, mi_predict_train, mi_labels, mi_labels, tag='MI ')
  print()
  print('   ### ' + RED + 'Mutual Information' + NC + ': Evaluation on ' + GREEN + 'Test Set' + NC)
  print_metrics(mi_labels_test, mi_predict_test, mi_labels, mi_labels, tag='MI ', outfile=outfile)
  
  ### Semantic Correlation
  print()
  print('Train Multiclass SVM for Semantic Correlation ...')
  #lin_clf_sc = LinearSVC(C=weight_decay, multi_class='crammer_singer')
  lin_clf_sc = LinearSVC(multi_class='crammer_singer')
  gdCV = GridSearchCV(lin_clf_sc, par_grid, n_jobs=4)
  gdCV.fit(features_train, sc_labels_train) 
  print('Best parameter found via grid search: ' + str(gdCV.best_params_))
  # store the best SC classifier for later usage
  with open('best_sc_svm.pickle', 'wb') as f:
    pickle.dump(gdCV, f)
  
  sc_predict_train = gdCV.predict(features_train)
  sc_predict_test = gdCV.predict(features_test)
  
  sc_labels = list(range(5))
  sc_labels_print = ((np.array(sc_labels) - 2) / 2.0).tolist()
  
  print()
  print('   ### ' + RED + 'Semantic Correlation' + NC + ': Evaluation on ' + GREEN + 'Training Set' + NC)
  print_metrics(sc_labels_train, sc_predict_train, sc_labels, sc_labels_print, tag='SC ')
  print()
  print('   ### ' + RED + 'Semantic Correlation' + NC + ': Evaluation on ' + GREEN + 'Test Set' + NC)
  print_metrics(sc_labels_test, sc_predict_test, sc_labels, sc_labels_print, tag='SC ', outfile=outfile)
  
  outfile.close()  
  
