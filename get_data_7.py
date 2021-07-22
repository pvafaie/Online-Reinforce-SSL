import numpy as np
import pandas as pd
import random
from skmultiflow.metrics.measure_collection import ClassificationMeasurements
from numpy.random import choice
from skmultiflow.trees import HoeffdingTreeClassifier
import sklearn.metrics as metrics
from sklearn.metrics import hamming_loss
# from IOB import *
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
# from Oza_bag import OzaBaggingClassifier_with_OB
from skmultiflow.meta import OzaBaggingClassifier
from skmultiflow.neural_networks import PerceptronMask
# from imblearn.over_sampling import SMOTE
# from semi_supervised_algorithms import SMOTE_OB_semi
from sklearn.linear_model import LogisticRegression




def get_data_NN(data, classes1, labels=None, warm_up=1000,
                              window=1, quantile=0.25):
    random.seed(0)
    np.random.seed(0)
    all_x = []
    all_y = []
    classes_precision = {}
    classes_recall = {}
    recalls_for_all_classes = {}
    instance_number = {}
    w = {}
    correct_instances_added = []
    incorrect_instances_added = []
    instance_number_self = []
    unlabeled_instances = []
    unlabeled_instances.append(0)
    classes_recall_forgetting_factor = {}
    instances_added = {}
    for value in classes1:
        w[value] = 0
        classes_precision[int(value)] = []
        recalls_for_all_classes[int(value)] = []
        instance_number[int(value)] = []
        recalls_for_all_classes[int(value)].append(0)
        instance_number[int(value)].append(0)
        classes_precision[int(value)].append(0)
        classes_precision[int(value)].append(0)
        instances_added[value] = []
        instances_added[value].append(0)
        instances_added[value].append(0)

    for value in classes1:
        classes_recall[int(value)] = []
        classes_recall[int(value)].append(0)
        classes_recall[int(value)].append(0)
    for value in classes1:
        classes_recall_forgetting_factor[int(value)] = 0
    measure = ClassificationMeasurements()
    warmup = False
    # if clf is None:
    #     warmup = True
    #     # clf = IOB_Classifier(HoeffdingTreeClassifier(), threshold=threshold_UOB, forgetting_factor=forgetting_factor, m=m)

    i = 0
    Xtmp = data[i:i + warm_up]
    X = []
    y = []
    x_prevwindow = []
    y_prevwindow = []
    for var in Xtmp:
        if not np.isnan(var[-1]):
            X.append(var[:-1])
            x_prevwindow.append(var[:-1])
            y_prevwindow.append(var[-1])
            y.append(int(var[-1]))
    X = np.array(X)
    y = np.array(y)
    # if warmup:
    #     # clf.partial_fit(X, y, classes1, warm_up=True)
    # else:
    #     # clf.partial_fit(X, y, classes1)
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(X)
    distances, indices = nbrs.kneighbors(X)
    mean_distances_warm_up = np.mean(distances)
    std_distances_warm_up = np.std(distances)
    eps = np.quantile(distances[:, 1], quantile)
    i += warm_up
    print(f"eps is {eps}")
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
            # print(var)
            if not np.isnan(var[-1]):
                X.append(var[:-1])

                x_prevwindow.append(var[:-1])
                y_prevwindow.append(int(var[-1]))
                x_prevwindow.pop(0)
                y_prevwindow.pop(0)
                # print(x_prevwindow)

                y.append(int(var[-1]))
                l += 1

            else:
                X_unlabled = var[:-1]
                Y_unlabled = y_labels[count]


                k += 1


                x_prevwindow_array = np.array(x_prevwindow.copy())
                # print(x_prevwindow_array)
                y_prevwindow_array = np.array(y_prevwindow.copy())

                neigh = NearestNeighbors(n_neighbors=10)
                NN_clf_5 = KNeighborsClassifier(n_neighbors=5)
                NN_clf_3 = KNeighborsClassifier(n_neighbors=3)
                NN_clf_7 = KNeighborsClassifier(n_neighbors=7)
                # print(x_prevwindow_array.shape)
                nbrs = neigh.fit(x_prevwindow_array)
                # print(y_prevwindow_array)
                NN_clf_3.fit(x_prevwindow_array, y_prevwindow_array)
                NN_clf_5.fit(x_prevwindow_array, y_prevwindow_array)
                NN_clf_7.fit(x_prevwindow_array, y_prevwindow_array)
                proba_NN_3 = NN_clf_3.predict_proba(np.array([X_unlabled]))
                pred_NN_3 = NN_clf_3.predict(np.array([X_unlabled]))
                proba_NN_5 = NN_clf_5.predict_proba(np.array([X_unlabled]))
                pred_NN_5 = NN_clf_5.predict(np.array([X_unlabled]))
                proba_NN_7 = NN_clf_7.predict_proba(np.array([X_unlabled]))
                pred_NN_7 = NN_clf_7.predict(np.array([X_unlabled]))
                # print(X_unlabled)
                distances, indices = nbrs.kneighbors(np.array([X_unlabled]))
                # print(f"all distances are {distances}")
                # print(f"all indices are {indices}")
                dirst_indi = zip(distances,indices)
                dirst_indi = sorted(dirst_indi,key =lambda pair: pair[0])
                indices_sorted =[x for _, x in dirst_indi]
                distances_sorted = [y for y, _ in dirst_indi]
                # print(distances)
                # print(f"all distances sorted {distances_sorted}")
                # print(f"all indices sorted {indices_sorted}")
#                 print(f"all distances {distances_sorted}")
#                 mean_distances_all = np.mean(distances_sorted)
#                 print(f"mean all {mean_distances_all}")
#                 std_distances_all = np.std(distances_sorted)
                distances_sorted = (distances_sorted - mean_distances_warm_up) / std_distances_warm_up
#                 print(f"nomalized distances {distances_sorted}")
                distance_NN = distances_sorted[0][0]
#                 print(f"nearest {distance_NN}")
                distance_NN_mean_3 = np.mean(distances_sorted[0][0:3])
#                 print(f"mean 3 {distance_NN_mean_3}")
                distance_NN_mean_5 = np.mean(distances_sorted[0][0:5])
                distance_NN_mean_7 = np.mean(distances_sorted[0][0:7])
#                 print(f"mean 5 {distance_NN_mean_5}")
                # preds = clf.predict(np.array([X_unlabled]))
                # probas = clf.predict_proba(np.array([X_unlabled]))
                add_y2 = []
                add_y1 = []
                # print(distances)
                # print(probas)
                features = np.dstack([
distance_NN,distance_NN_mean_3, np.max(proba_NN_3, axis=1),
 (pred_NN_3==y_prevwindow_array[indices_sorted[0][0]]).astype(int),distance_NN_mean_5,np.max(proba_NN_5, axis=1),
(pred_NN_3 == pred_NN_5).astype(int),distance_NN_mean_7,np.max(proba_NN_7, axis=1),(pred_NN_5 == pred_NN_7).astype(int)])
                # print(features[0])
                # critic_predition = critic.predict(features[0])
                # print(features[0])
                all_x.append(features[0][0])
#                 print(f"features are {features[0][0]}")
                if y_prevwindow_array[indices_sorted[0][0]] == pred_NN_5[0] == pred_NN_3[0] == pred_NN_7 == Y_unlabled:
                    # print(y_prevwindow_array[indices_sorted[0][0]])
                    # print(pred_NN_5[0])
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

#                 print(f"y is {all_y[-1]}")


                # x_prevwindow.append(X_unlabled)
                # #         # print(f"yprev is {y_prevwindow_array.astype(int)[indices][0][0]}")
                # y_prevwindow.append( Y_unlabled)
                # x_prevwindow.pop(0)
                # y_prevwindow.pop(0)




#                 k = 0
#                 l = 0
#                 X_unlabled = []
#                 Y_unlabled = []


#                 correct_instances_added.append(sum([v[0] for v in list(instances_added.values())]))
#                 incorrect_instances_added.append(sum([v[1] for v in list(instances_added.values())]))
#                 instance_number_self.append(i)

#         if len(X) == 0:
#             i = j
#             continue

        i = j
        print(f"{i} out of {len(data)}", end="\r")



    return  np.array(all_x), np.array(all_y)


def get_data_experiment_NN(ALL_DATA, ALL_CLASSES, missing_rates):
    X = []
    y = []
    for (classes, data) in zip(ALL_CLASSES, ALL_DATA):
        for missing_rate in missing_rates:
            np.random.seed(42)
            random.seed(42)
            data_semi = data.astype(float)
            labels = data.astype(float)[:, -1]
            data_semi[np.random.choice(np.arange(1000, len(data)), int(len(data_semi[1000:]) * missing_rate), replace=False), -1] = np.NaN
            # print(data_semi)
            random.seed(42)
            np.random.seed(42)
            print(f"Missing_rate: {missing_rate}")
            X_d, y_d = get_data_NN(data_semi, classes, labels=labels)
            X.append(X_d)
            y.append(y_d)

    return np.array(X), np.array(y)


def experiment_all_every_instance(data, classes1, missing_rates, semi_approach='proba', m=10,
                                          threshold_UOB=0.05, forgetting_factor=0.9, warm_up=1000, quantile=0.25,
                                          critic=None,clf = None):
    np.random.seed(10)
    random.seed(10)
    results = []
    experiment_name = []
    for missing_rate in missing_rates:
        np.random.seed(10)
        random.seed(10)
        data_semi = data.astype(float)
        labels = data.astype(float)[:, -1]
        data_semi[np.random.choice(np.arange(1000,len(data)), int(len(data_semi[1000:]) * missing_rate), replace=False), -1] = np.NaN
        # print(data_semi)
        random.seed(10)
        np.random.seed(10)
        print(f"Missing_rate: {missing_rate}")
        results.append(get_data_NN(data_semi, classes1, labels=labels,
                                                 warm_up=warm_up, quantile=quantile))
        #             experiment_name.append(f"Missing_rate: {missing_rate}, threshhold:{thresh_hold}")
        print("***************************")
    return results


