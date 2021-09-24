import numpy as np
import pandas as pd
import random
from skmultiflow.metrics.measure_collection import ClassificationMeasurements
from numpy.random import choice
from skmultiflow.trees import HoeffdingTreeClassifier
import sklearn.metrics as metrics
from sklearn.metrics import hamming_loss
from IOE import IOE_Classifier
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
from skmultiflow.meta import OzaBaggingClassifier
from skmultiflow.neural_networks import PerceptronMask
from sklearn.linear_model import LogisticRegression




def Meta_reinforce(data, classes1,  labels=None, warm_up=500, 
                                critic=None,clf=None):
    random.seed(0)
    np.random.seed(0)
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
    if clf is None:
        warmup = True
        clf = IOE_Classifier(HoeffdingTreeClassifier())

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
    if warmup:
        clf.partial_fit(X, y, classes1, warm_up=True)
    else:
        clf.partial_fit(X, y, classes1)
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(X)
    distances, indices = nbrs.kneighbors(X)
    eps = np.quantile(distances[:, 1], quantile)
    i += warm_up
    print(f"eps is {eps}")
    k = 0
    l = 0
    X_unlabled = []
    Y_unlabled = []
    counts = {}
    counts[1] = 0
    counts[2] = 0
    counts[3] = 0

    while (i + window < len(data)):
        j = i + window
        Xtmp = data[i:j]
        y_labels = labels[i:j]
        X = []
        y = []
        for (count, var) in enumerate(Xtmp):
            # print(var)
            result = clf.predict(np.array([var[:-1]]))
            result = result[0]
            measure.add_result(y_true=y_labels[count], y_pred=result, weight=1.0)
            if (y_labels[count] == result):
                classes_precision[result][0] += 1
                classes_recall[result][0] += 1
            else:
                if result in classes1:
                    classes_precision[result][1] += 1
                else:
                    print(f"{result} not in classes")
                classes_recall[y_labels[count]][1] += 1
            for key, re in classes_recall.items():
                instance_number[key].append(i)
                if (re[0] + re[1]) != 0:
                    recalls_for_all_classes[key].append(re[0] / (re[0] + re[1]))
                else:
                    recalls_for_all_classes[key].append(0)
            if not np.isnan(var[-1]):
                X.append(var[:-1])
                # print(X)
                x_prevwindow.append(var[:-1])
                y_prevwindow.append(int(var[-1]))
                x_prevwindow.pop(0)
                y_prevwindow.pop(0)
                y.append(int(var[-1]))
                l += 1

            else:
                k += 1
                X_unlabled = var[:-1]
                Y_unlabled = y_labels[count]          
                x_prevwindow_array = np.array(x_prevwindow.copy())
                y_prevwindow_array = np.array(y_prevwindow.copy())
                neigh = NearestNeighbors(n_neighbors=10)
                NN_clf_5 = KNeighborsClassifier(n_neighbors=5)
                NN_clf_3 = KNeighborsClassifier(n_neighbors=3)
                nbrs = neigh.fit(x_prevwindow_array)
                NN_clf_3.fit(x_prevwindow_array, y_prevwindow_array)
                NN_clf_5.fit(x_prevwindow_array, y_prevwindow_array)
                proba_NN_3 = NN_clf_3.predict_proba(np.array([X_unlabled]))
                pred_NN_3 = NN_clf_3.predict(np.array([X_unlabled]))
                proba_NN_5 = NN_clf_5.predict_proba(np.array([X_unlabled]))
                pred_NN_5 = NN_clf_5.predict(np.array([X_unlabled]))
                distances, indices = nbrs.kneighbors(np.array([X_unlabled]))
                dirst_indi = zip(distances, indices)
                dirst_indi = sorted(dirst_indi, key=lambda pair: pair[0])
                indices_sorted = [x for _, x in dirst_indi]
                distances_sorted = [y for y, _ in dirst_indi]
                mean_distances_all = np.mean(distances_sorted)
                std_distances_all = np.std(distances_sorted)
                distances_sorted = (distances_sorted - mean_distances_all) / std_distances_all
                distance_NN = distances_sorted[0][0]
                distance_NN_mean_3 = np.mean(distances_sorted[0][0:3])
                distance_NN_mean_5 = np.mean(distances_sorted[0][0:5])
                add_y2 = []
                add_y1 = []
                features = np.dstack([
                    distance_NN, distance_NN_mean_3, np.max(proba_NN_3, axis=1),
                    (pred_NN_3 == y_prevwindow_array[indices_sorted[0][0]]).astype(int), distance_NN_mean_5,
                    np.max(proba_NN_5, axis=1),
                    (pred_NN_3 == pred_NN_5).astype(int)])
                critic_predition = critic.predict(features[0])[0]
                if np.count_nonzero(critic_predition == 1) > 0:
                    counts[1]+=1
                    if len(X) > 0:
                        X = np.vstack((X, np.array([X_unlabled])))
                        print( y_prevwindow_array.astype(int)[indices_sorted])
                        y = np.hstack((y, y_prevwindow_array.astype(int)[indices_sorted]))
                        print(y)
                    else:
                        X = np.array([X_unlabled])
                        y = np.array([y_prevwindow_array.astype(int)[indices_sorted[0][0]]])                    

                if np.count_nonzero(critic_predition == 2) > 0:
                    counts[2]+=1
                    if len(X)>0:
                        X = np.vstack((X, np.array([X_unlabled])))
                        y = np.hstack((y, pred_NN_3))
                    else:
                        X = np.array([X_unlabled])
                        y =  pred_NN_3

                if np.count_nonzero(critic_predition == 3) > 0:
                    counts[3]+=1
                    if len(X)>0:
                        X = np.vstack((X, np.array([X_unlabled])))
                        y = np.hstack((y, pred_NN_5))
                    else:
                        X = np.array([X_unlabled])
                        y =  pred_NN_5

                true_labels2 = Y_unlabled
                if np.count_nonzero(critic_predition == 1) > 0:
                    if true_labels2 ==y_prevwindow_array.astype(int)[indices_sorted[0][0]]:
                        instances_added[true_labels2][0] += 1
                    else:
                        instances_added[true_labels2][1] += 1
                        
                if np.count_nonzero(critic_predition == 2) > 0:
                    if true_labels2 ==pred_NN_3:
                        instances_added[true_labels2][0] += 1
                    else:
                        instances_added[true_labels2][1] += 1
                        
                if np.count_nonzero(critic_predition == 3) > 0:
                    if true_labels2 == pred_NN_5:
                        instances_added[true_labels2][0] += 1
                    else:
                        instances_added[true_labels2][1] += 1

                k = 0
                l = 0
                X_unlabled = []
                Y_unlabled = []
                correct_instances_added.append(sum([v[0] for v in list(instances_added.values())]))
                incorrect_instances_added.append(sum([v[1] for v in list(instances_added.values())]))
                instance_number_self.append(i)

        if len(X) == 0:
            i = j
            continue
        X = np.array(X)
        y = np.array(y)
        clf.partial_fit(X, y, classes=classes1)
        i = j
        print(f"{i} out of {len(data)}", end="\r")

    Final_result = []
    Final_result.append(measure.get_accuracy())
    Final_result.append(measure.get_kappa())
    Final_result.append(measure.get_kappa_m())
    Final_result.append(measure.get_kappa_t())
    Final_result.append(classes_recall.items())
    print(w)
    print(f"Finished")
    print(f"Final Acc is {measure.get_accuracy()}")
    print(f"Final Kappa is {measure.get_kappa()}")
    print(f"Final Kappa_M is {measure.get_kappa_m()}")
    print(f"Final Kappa_T is {measure.get_kappa_t()}")
    print(f"Recall is {measure.get_recall()}")
    print(f"Precision is {measure.get_precision()}")
    print(f"count NN is {counts[1]}")
    print(f"count 3NN is {counts[2]}")
    print(f"count 5NN is {counts[3]}")
    recall = 1
    recalls = []
    precisions = []
    macro_recall = 0
    macro_precision = 0
    for key, var in instances_added.items():
        if (var[0] + var[1]) != 0:
            print(f"instances correctly added to the  class {key} are {var[0]} out of {var[0] + var[1]}")
        else:
            print(f"0 instances added to the class {key}")
    for key, var in classes_recall.items():
        if (var[0] + var[1]) != 0:
            recall *= (var[0] / (var[0] + var[1]))
            print(f"class {str(key)} recall : {str(var[0] / (var[0] + var[1]))} ")
            print(var[0] + var[1])
            recalls.append((var[0] / (var[0] + var[1])))
            macro_recall += (var[0] / (var[0] + var[1]))
    print(f"macro recall is {macro_recall / len(classes1)}")
    for key, var in classes_precision.items():
        #         recall*=(var[0]/( var[0]+var[1]))
        if (var[0] + var[1]) != 0:
            print(f"class {str(key)} precision : {str(var[0] / (var[0] + var[1]))} ")
            macro_precision += (var[0] / (var[0] + var[1]))
            precisions.append((var[0] / (var[0] + var[1])))
        else:
            precisions.append(0)
    print(f"macro precision is {macro_precision / len(classes1)}")
    macro_f1 = 0
    for i in range(len(recalls)):
        if precisions[i] + recalls[i] != 0:
            macro_f1 += 2 * recalls[i] * precisions[i] / (precisions[i] + recalls[i])
    print(f"macro_f1 is {macro_f1 / len(recalls)}")
    Final_result.append(recalls)
    Final_result.append(recalls_for_all_classes)
    Final_result.append(instance_number)
    print(f"G_mean {recall ** (1 / len(recalls))}")
    Final_result.append(recall ** (1 / len(recalls)))
    Final_result.append(unlabeled_instances)
    Final_result.append(correct_instances_added)
    Final_result.append(incorrect_instances_added)
    Final_result.append(instance_number_self)
    return Final_result


def Meta_reinforce_all_missing_rates(data, classes1, missing_rates,  critic=None,clf = None):
    np.random.seed(0)
    random.seed(0)
    results = []
    experiment_name = []
    for missing_rate in missing_rates:
        np.random.seed(0)
        random.seed(0)
        data_semi = data.astype(float)
        labels = data.astype(float)[:, -1]
        data_semi[np.random.choice(np.arange(1000,len(data)), int(len(data_semi[1000:]) * missing_rate), replace=False), -1] = np.NaN
        random.seed(0)
        np.random.seed(0)
        print(f"Missing_rate: {missing_rate}")
        results.append(meta_reinforce(data_semi, classes1, labels=labels, critic=critic,clf = clf))
        print("***************************")
    return results


