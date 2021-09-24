import numpy as np
import pandas as pd
import random
from skmultiflow.metrics.measure_collection import ClassificationMeasurements
from numpy.random import choice
from skmultiflow.trees import HoeffdingTreeClassifier
import sklearn.metrics as metrics
from sklearn.metrics import hamming_loss
from IOB import IOB_Classifier
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




def NN_with_everyinstance(data, classes1, threshold=0.91, semi_approach="proba", m=10,
                              threshold_UOB=0.05, forgetting_factor=0.9, labels=None, warm_up=500, window_1NN=500,
                              window=1, quantile=0.25, critic=None,clf=None):
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
        clf = IOB_Classifier(HoeffdingTreeClassifier(), threshold=threshold_UOB, forgetting_factor=forgetting_factor, m=m)

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
                # print(x_prevwindow)

                y.append(int(var[-1]))
                l += 1

            else:
                k += 1
                X_unlabled = var[:-1]
                Y_unlabled = y_labels[count]
               
                x_prevwindow_array = np.array(x_prevwindow.copy())
                # print(x_prevwindow_array)
                y_prevwindow_array = np.array(y_prevwindow.copy())

                neigh = NearestNeighbors(n_neighbors=10)
                NN_clf_5 = KNeighborsClassifier(n_neighbors=5)
                NN_clf_3 = KNeighborsClassifier(n_neighbors=3)
                # print(x_prevwindow_array.shape)
                nbrs = neigh.fit(x_prevwindow_array)
                # print(y_prevwindow_array)
                NN_clf_3.fit(x_prevwindow_array, y_prevwindow_array)
                NN_clf_5.fit(x_prevwindow_array, y_prevwindow_array)
                proba_NN_3 = NN_clf_3.predict_proba(np.array([X_unlabled]))
                pred_NN_3 = NN_clf_3.predict(np.array([X_unlabled]))
                proba_NN_5 = NN_clf_5.predict_proba(np.array([X_unlabled]))
                pred_NN_5 = NN_clf_5.predict(np.array([X_unlabled]))
                # print(X_unlabled)
                distances, indices = nbrs.kneighbors(np.array([X_unlabled]))
                # print(f"all distances are {distances}")
                # print(f"all indices are {indices}")
                dirst_indi = zip(distances, indices)
                dirst_indi = sorted(dirst_indi, key=lambda pair: pair[0])
                indices_sorted = [x for _, x in dirst_indi]
                distances_sorted = [y for y, _ in dirst_indi]
                # print(distances)
                # print(f"all distances sorted {distances_sorted}")
                # print(f"all indices sorted {indices_sorted}")
                mean_distances_all = np.mean(distances_sorted)
                # print(f"mean all {mean_distances_all}")
                std_distances_all = np.std(distances_sorted)
                distances_sorted = (distances_sorted - mean_distances_all) / std_distances_all
                # print(f"nomalized distances {distances_sorted}")
                distance_NN = distances_sorted[0][0]
                # print(f"nearest {distance_NN}")
                distance_NN_mean_3 = np.mean(distances_sorted[0][0:3])
                # print(f"nearest {distance_NN_mean_3}")
                distance_NN_mean_5 = np.mean(distances_sorted[0][0:5])

                add_y2 = []
                add_y1 = []
                # print(distances)
                # print(probas)
                features = np.dstack([
                    distance_NN, distance_NN_mean_3, np.max(proba_NN_3, axis=1),
                    (pred_NN_3 == y_prevwindow_array[indices_sorted[0][0]]).astype(int), distance_NN_mean_5,
                    np.max(proba_NN_5, axis=1),
                    (pred_NN_3 == pred_NN_5).astype(int)])
                # print(features[0])
                critic_predition = critic.predict(features[0])[0]
                # print(critic_predition)
                if np.count_nonzero(critic_predition == 1) > 0:
                    counts[1]+=1
                    # add_x2 = X_unlabled_array
                    # print(y_prevwindow_array)
                    # add_y2 = y_prevwindow_array.astype(int)[indices][0]
                    if len(X) > 0:
                        X = np.vstack((X, np.array([X_unlabled])))
                        print( y_prevwindow_array.astype(int)[indices_sorted])
                        y = np.hstack((y, y_prevwindow_array.astype(int)[indices_sorted]))
                        print(y)
                    else:
                        # print(y)
                        # print(indices_sorted)
                        # print( y_prevwindow_array.astype(int)[indices_sorted[0][0]])
                        X = np.array([X_unlabled])
                        y = np.array([y_prevwindow_array.astype(int)[indices_sorted[0][0]]])
                        # print(f"y is {y}")
                        # print(y)
                        # print(f"X_unlabled is {X_unlabled}")
                        # x_prevwindow.append(X_unlabled)
                        # # print(f"yprev is {y_prevwindow_array.astype(int)[indices][0][0]}")
                        # y_prevwindow.append(y_prevwindow_array.astype(int)[distances_sorted][0][0])
                        # x_prevwindow.pop(0)
                        # y_prevwindow.pop(0)






                if np.count_nonzero(critic_predition == 2) > 0:
                    # add_x2 = X_unlabled_array
                    # print(y_prevwindow_array)
                    counts[2]+=1
                    # add_y2 = y_prevwindow_array.astype(int)[indices][0]
                    if len(X)>0:
                        X = np.vstack((X, np.array([X_unlabled])))
                        y = np.hstack((y, pred_NN_3))
                    else:
                        # print(y)
                        X = np.array([X_unlabled])
                        y =  pred_NN_3
                        # print(y)
                        # print(f"X_unlabled is {X_unlabled}")
                        # x_prevwindow.append(X_unlabled)
                        # # print(f"yprev is {y_prevwindow_array.astype(int)[indices][0][0]}")
                        # y_prevwindow.append( pred_NN_3[0])
                        # x_prevwindow.pop(0)
                        # y_prevwindow.pop(0)


                if np.count_nonzero(critic_predition == 3) > 0:
                    # add_x2 = X_unlabled_array
                    # print(y_prevwindow_array)
                    # add_y2 = y_prevwindow_array.astype(int)[indices][0]
                    counts[3]+=1
                    if len(X)>0:
                        X = np.vstack((X, np.array([X_unlabled])))
                        y = np.hstack((y, pred_NN_5))
                    else:
                        # print(y)
                        X = np.array([X_unlabled])
                        y =  pred_NN_5
                        # print(y)
                        # print(f"X_unlabled is {X_unlabled}")
                        # x_prevwindow.append(X_unlabled)
                        # # print(f"yprev is {y_prevwindow_array.astype(int)[indices][0][0]}")
                        # y_prevwindow.append( pred_NN_5[0])
                        # x_prevwindow.pop(0)
                        # y_prevwindow.pop(0)


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

                # print(features)

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
        # print(X.shape)
        # print(y.shape)
        # print(y)
        # result = clf.predict(X)
        # probas = clf.predict_proba(X)
        # precisions = {}
        # recalls = {}
        # for value in classes1:
        #     precisions[int(value)] = []
        #     precisions[int(value)].append(0)
        #     precisions[int(value)].append(0)

        # for value in classes1:
        #     recalls[int(value)] = []
        #     recalls[int(value)].append(0)
        #     recalls[int(value)].append(0)

        # for m in range(0, len(result)):

        #     measure.add_result(y_true=y[m], y_pred=result[m], weight=1.0)
        #     if (y[m] == result[m]):
        #         classes_precision[y[m]][0] += 1
        #         classes_recall[y[m]][0] += 1
        #         recalls[y[m]][0] += 1
        #     else:
        #         if result[m] in classes1:
        #             classes_precision[result[m]][1] += 1
        #         else:
        #             print(f"{result[m]} not in classes")
        #         classes_recall[y[m]][1] += 1
        #         recalls[y[m]][1] += 1
        # for key, var in w.items():
        #     w[key] = forgetting_factor * w[key] + (1 - forgetting_factor) * (np.count_nonzero(y == key))
        # for key, var in classes_recall.items():
        #     instance_number[key].append(i)
        #     if (var[0] + var[1]) != 0:
        #         recalls_for_all_classes[key].append(var[0] / (var[0] + var[1]))
        #     else:
        #         recalls_for_all_classes[key].append(0)

        # if k > window_1NN and l > window_1NN:

        #     labels_for_data = []
        #     unlabel_count = unlabeled_instances[-1]
        #     unlabeled_instances.append(unlabel_count + len(X_unlabled))

        # print(X.shape)
        # print(X)
        clf.partial_fit(X, y, classes=classes1)

        i = j
        print(f"{i} out of {len(data)}", end="\r")

    if (i < len(data)):
        Xtmp = data[i:len(data)]
        X = []
        y = []
        for var in Xtmp:
            if not np.isnan(var[-1]):
                X.append(var[:-1])
                y.append(int(var[-1]))
            else:
                X_unlabled.append(var[:-1])
                k += 1
        if len(X) != 0:
            result = clf.predict(X)
            for m in range(0, len(result)):
                measure.add_result(y_true=y[m], y_pred=result[m], weight=1.0)
                if (y[m] == result[m]):
                    classes_recall[y[m]][0] += 1
                    classes_precision[y[m]][0] += 1
                else:
                    classes_precision[result[m]][1] += 1
                    classes_recall[y[m]][1] += 1
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


def experiment_all_NN_every_instance(data, classes1, missing_rates, semi_approach='proba', m=10,
                                          threshold_UOB=0.05, forgetting_factor=0.9, warm_up=1000, quantile=0.25,
                                          critic=None,clf = None):
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
        # print(data_semi)
        random.seed(0)
        np.random.seed(0)
        print(f"Missing_rate: {missing_rate}")
        results.append(NN_with_everyinstance(data_semi, classes1, semi_approach=semi_approach, labels=labels, m=m,
                                                 threshold_UOB=threshold_UOB, forgetting_factor=forgetting_factor,
                                                 warm_up=warm_up, quantile=quantile, critic=critic,clf = clf))
        #             experiment_name.append(f"Missing_rate: {missing_rate}, threshhold:{thresh_hold}")
        print("***************************")
    return results


