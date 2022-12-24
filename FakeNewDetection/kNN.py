import numpy as np
import math


def euclidean(point, data):
    return np.sqrt(np.sum((point - data)**2, axis=1))


def get_common_label(list_label):
    return max(set(list_label), key=list_label.count)


class my_kNN:
    def __init__(self, k=1, distance_metrics=euclidean):
        self.k = k
        self.distance_metrics = distance_metrics
        self.x_train = []
        self.y_train = []

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    def predict(self, x_test):
        # result = []
        # for index, raw_predict in x.iterrows():
        # # for raw in x:
        #     list_distance = []
        #     for index_train, raw_train in self.x_train.iterrows():
        #         distance = self.__calculate_distance(raw_predict, raw_train)
        #         label = y_train[index_train]
        #         list_distance.append((distance, label))
        #     c = self.__get_label(list_distance)
        #     result.append(c)
        # return result
        result = []
        # for index, point in x_test.iterrows():
        for point in x_test:
            distances = self.distance_metrics(point, self.x_train)
            k_neareast_label = [y for _, y in sorted(zip(distances, self.y_train))]
            result.append(k_neareast_label[:self.k])
        return list(map(get_common_label, result))

    def __calculate_distance(self, point_a, point_b):
        tmp = 0
        for i in range(len(point_a)):
            tmp += (float(point_a[i]) - float(point_b[i])) ** 2
        return math.sqrt(tmp)

    def __get_label(self, list_distance):
        sorted(list_distance, key=lambda tup: tup[0])
        k_neareast = list_distance[:self.k]
        count_by_label = dict()
        for value in k_neareast:
            if value[1] in count_by_label:
                count_by_label[value[1]] += 1
            else:
                count_by_label[value[1]] = 1
        return max(count_by_label, key=count_by_label.get)
