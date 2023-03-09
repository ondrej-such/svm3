
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import numpy as np


def fit_liblinear(x, y, multi_class="ovr"):
    clf = make_pipeline(StandardScaler(),
                        LinearSVC(random_state=0, tol=1e-3, max_iter = 100000, multi_class=multi_class))
    params = {
                'linearsvc__max_iter' : 100000,
                    'linearsvc__tol' : 1e-3
                    }

    clf.set_params(**params)
    mod = clf.fit(x, y)
    return(np.sum(np.equal(clf.predict(x) , y)) / len(y))
