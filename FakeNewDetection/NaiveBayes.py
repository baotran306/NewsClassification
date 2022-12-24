import numpy as np
import pandas as pd
import math


class MyGaussianNB:
    def __init__(self):
        self.classes = []
        self.prior_probs = {}
        self.mean = {}
        self.stds = {}

    def fit(self, x, y):
        sample = len(y)  # <=> X.shape[0]
        self.classes, counts = np.unique(y, return_counts=True)
        print(self.classes)
        self.prior_probs = dict(zip(self.classes, counts/sample))
        self.mean = {}
        self.stds = {}

        for c in self.classes:
            idx = np.argwhere(y.to_numpy() == c)  # return index of class cl in y_data
            x_with_class_c = x[idx, :]
            # x_with_class_c = x.loc[y == c]
            self.mean[c] = np.mean(x_with_class_c, axis=0).flatten()#.tolist()#
            self.stds[c] = np.std(x_with_class_c, ddof=1, axis=0).flatten()#.tolist()#
            # print(self.mean[c])
            # print(self.stds[c])

    def predict(self, x):
        result = []
        # for index, raw in x.iterrows():
        for raw in x:
            class_probs = self.__compute_class_probs(raw)
            c = max(class_probs, key=class_probs.get)
            result.append(c)
        return result

    def __compute_class_probs(self, x):
        p_of_class = {}
        for c in self.classes:
            p_of_class[c] = self.prior_probs[c]
            for i, v in enumerate(x):
                p_of_class[c] *= self.__norm_pdf(v, self.mean[c][i], self.stds[c][i])
        return p_of_class

    def __norm_pdf(self, x, mean, std):
        exp = np.exp(-((x - mean) ** 2 / (2 * std ** 2)))
        return (1 / (std * np.sqrt(2 * np.pi))) * exp
