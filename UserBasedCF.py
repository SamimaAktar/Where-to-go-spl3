import numpy as np
from numpy.linalg import norm
import time


class UserBasedCF(object):
    def __init__(self):
        self.rec_score = None
        self.C_ = None
        self.norms_ = None

    def load_result(self, path):
        ctime = time.time()
        print("Loading result...",)
        self.rec_score = np.load(path + "rec_score.npy")
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def save_result(self, path):
        ctime = time.time()
        print("Saving result...",)
        np.save(path + "rec_score", self.rec_score)
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def pre_compute_rec_scores(self, C):
        ctime = time.time()
        self.C_ = C
        print("Training User-based Collaborative Filtering...",C.shape )

        sim = C.dot(C.T)
        norms = np.array([norm(C[i]) for i in range(C.shape[0])])
        self.norms_ = norms
        sim = sim/(norms.reshape((-1,1)) * norms.reshape((1,-1)))
        np.fill_diagonal(sim,0.0)
        self.rec_score = sim.dot(C)
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def predict_user(self, X):
        ctime = time.perf_counter()
        sim = X.dot(self.C_.T)
        print("S:",sim.shape)
        norms = np.array([norm(X[i]) for i in range(X.shape[0])])
        boo = (norms.reshape((-1,1)) * self.norms_.reshape((1,-1)))
        print(boo.shape)
        sim = sim/boo
        np.fill_diagonal(sim,0.0)
        rec_score = sim.dot(self.C_)
        print("Done. Elapsed time:", time.perf_counter() - ctime, "s",rec_score)
        return rec_score

    def predict(self, i, j):
        return self.rec_score[i][j]
