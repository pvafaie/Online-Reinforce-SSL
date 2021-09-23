import numpy as np
import pandas as pd
import random
from skmultiflow.metrics.measure_collection import ClassificationMeasurements
from numpy.random import choice
from skmultiflow.trees import HoeffdingTreeClassifier
import sklearn.metrics as metrics
from sklearn.metrics import hamming_loss
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
from skmultiflow.neural_networks import PerceptronMask
from sklearn.linear_model import LogisticRegression

# This function is used for preparing the data needed for training the meta-RL model. The get_data_with_missing_rates function is used for iterating through all the missing rates and creating a full meta-dataset. The get_data is used for each individual missing rate. 


def get_data(data, classes, labels=None, warm_up=1000,
                              window=1, quantile=0.25):
    random.seed(0)
    np.random.seed(0)
    all_x = []
    all_y = []
    instance_number = {}
    w = {}
    warmup = False
    i = 0
    Xtmp = data[i:i + warm_up]
    X = []
    y = []
    x_prevwindow = []
    y_prevwindow = []
#     add warm-up data
    for var in Xtmp:
        if not np.isnan(var[-1]):
            X.append(var[:-1])
            x_prevwindow.append(var[:-1])
            y_prevwindow.append(var[-1])
            y.append(int(var[-1]))
    X = np.array(X)
    y = np.array(y)
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(X)
    distances, indices = nbrs.kneighbors(X)
    mean_distances_warm_up = np.mean(distances)
    std_distances_warm_up = np.std(distances)
    i += warm_up
    k = 0
    l = 0
    X_unlabled = []
    Y_unlabled = []


    while (i + window < len(data)):
        j = i + window
        Xtmp = data[i:j]
        y_labels = labels[i:j]
        X = []
        y = []
        for (count, var) in enumerate(Xtmp):
#             if instance is labelled, add it to the window
            if not np.isnan(var[-1]):
                X.append(var[:-1])
                x_prevwindow.append(var[:-1])
                y_prevwindow.append(int(var[-1]))
                x_prevwindow.pop(0)
                y_prevwindow.pop(0)
                y.append(int(var[-1]))
                l += 1

            else:
#                 if instance is not labeled, then get the KNN predictions. The aim here is to see which KNN classifiers have the correct prediction. 
                X_unlabled = var[:-1]
                Y_unlabled = y_labels[count]
                k += 1
                x_prevwindow_array = np.array(x_prevwindow.copy())
                y_prevwindow_array = np.array(y_prevwindow.copy())
                neigh = NearestNeighbors(n_neighbors=10)
                NN_clf_5 = KNeighborsClassifier(n_neighbors=5)
                NN_clf_3 = KNeighborsClassifier(n_neighbors=3)
                NN_clf_7 = KNeighborsClassifier(n_neighbors=7)
                nbrs = neigh.fit(x_prevwindow_array)
                NN_clf_3.fit(x_prevwindow_array, y_prevwindow_array)
                NN_clf_5.fit(x_prevwindow_array, y_prevwindow_array)
                NN_clf_7.fit(x_prevwindow_array, y_prevwindow_array)
                proba_NN_3 = NN_clf_3.predict_proba(np.array([X_unlabled]))
                pred_NN_3 = NN_clf_3.predict(np.array([X_unlabled]))
                proba_NN_5 = NN_clf_5.predict_proba(np.array([X_unlabled]))
                pred_NN_5 = NN_clf_5.predict(np.array([X_unlabled]))
                proba_NN_7 = NN_clf_7.predict_proba(np.array([X_unlabled]))
                pred_NN_7 = NN_clf_7.predict(np.array([X_unlabled]))
                distances, indices = nbrs.kneighbors(np.array([X_unlabled]))
                dirst_indi = zip(distances,indices)
                dirst_indi = sorted(dirst_indi,key =lambda pair: pair[0])
                indices_sorted =[x for _, x in dirst_indi]
                distances_sorted = [y for y, _ in dirst_indi]
                distances_sorted = (distances_sorted - mean_distances_warm_up) / std_distances_warm_up
                distance_NN = distances_sorted[0][0]
                distance_NN_mean_3 = np.mean(distances_sorted[0][0:3])
                distance_NN_mean_5 = np.mean(distances_sorted[0][0:5])
                distance_NN_mean_7 = np.mean(distances_sorted[0][0:7])
                add_y2 = []
                add_y1 = []
                features = np.dstack([
distance_NN,distance_NN_mean_3, np.max(proba_NN_3, axis=1),
 (pred_NN_3==y_prevwindow_array[indices_sorted[0][0]]).astype(int),distance_NN_mean_5,np.max(proba_NN_5, axis=1),
(pred_NN_3 == pred_NN_5).astype(int),distance_NN_mean_7,np.max(proba_NN_7, axis=1),(pred_NN_5 == pred_NN_7).astype(int)])
                all_x.append(features[0][0])
                
#                 Check which combination the label is. For example, if all the KNN predictions are correct, then the combination is number 16, and the label 16 is added. In another case, if three of them are correct, then other combinations are selected as the label. These labels will be used for training the meta-RL model. 
                if y_prevwindow_array[indices_sorted[0][0]] == pred_NN_5[0] == pred_NN_3[0] == pred_NN_7 == Y_unlabled:
                    all_y.append(16)
                elif y_prevwindow_array[indices_sorted[0][0]] == pred_NN_3[0] == pred_NN_5[0]==Y_unlabled:
                    all_y.append(15)
                elif y_prevwindow_array[indices_sorted[0][0]] == pred_NN_3[0] == pred_NN_7[0]==Y_unlabled:
                    all_y.append(14)
                elif y_prevwindow_array[indices_sorted[0][0]] == pred_NN_5[0] == pred_NN_7[0]==Y_unlabled:
                    all_y.append(13)
                elif pred_NN_3[0] == pred_NN_5[0] == pred_NN_7[0]==Y_unlabled:
                    all_y.append(12)
                elif pred_NN_5[0] == pred_NN_7[0]==Y_unlabled:
                    all_y.append(11) 
                elif pred_NN_3[0] == pred_NN_7[0]==Y_unlabled:
                    all_y.append(10) 
                elif y_prevwindow_array[indices_sorted[0][0]] == pred_NN_7[0]==Y_unlabled:
                    all_y.append(9)
                elif pred_NN_3[0] == pred_NN_5[0]==Y_unlabled:
                    all_y.append(8)
                elif y_prevwindow_array[indices_sorted[0][0]] == pred_NN_5[0]==Y_unlabled:
                    all_y.append(7)
                elif y_prevwindow_array[indices_sorted[0][0]] == pred_NN_3[0]==Y_unlabled:
                    all_y.append(6)
                elif pred_NN_7[0]==Y_unlabled:
                    all_y.append(5)
                elif pred_NN_5[0]==Y_unlabled:
                    all_y.append(4)
                elif pred_NN_3[0]==Y_unlabled:
                    all_y.append(3)
                elif  y_prevwindow_array[indices_sorted[0][0]]==Y_unlabled:
                    all_y.append(2)
                else:
                    all_y.append(1)

        i = j
        print(f"{i} out of {len(data)}", end="\r")
    return  np.array(all_x), np.array(all_y)


def get_data_with_missing_rates(ALL_DATA, ALL_CLASSES, missing_rates):
    X = []
    y = []
    for (classes, data) in zip(ALL_CLASSES, ALL_DATA):
        for missing_rate in missing_rates:
            np.random.seed(42)
            random.seed(42)
            data_semi = data.astype(float)
            labels = data.astype(float)[:, -1]
            data_semi[np.random.choice(np.arange(1000, len(data)), int(len(data_semi[1000:]) * missing_rate), replace=False), -1] = np.NaN
            random.seed(42)
            np.random.seed(42)
            print(f"Missing_rate: {missing_rate}")
            X_d, y_d = get_data(data_semi, classes, labels=labels)
            X.append(X_d)
            y.append(y_d)

    return np.array(X), np.array(y)



