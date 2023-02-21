from functools import reduce
from cmath import *
import pandas as pd
import random
import time
import numpy as np

class Polynomial:
    def __init__(self, coeffs):
        self.coeffs = coeffs
        self.deg = len(coeffs) - 1

    def __str__(self):
        return "Poly({})".format(self.coeffs)

    def __call__(self, x):
        return reduce(lambda r, a: r * x + a, reversed(self.coeffs[:self.deg]), self.coeffs[-1])

    def __mul__(self, Q):
        n = self.deg + Q.deg + 1
        (p, q) = (self.get(self.coeffs), self.get(Q.coeffs))
        return Polynomial([sum([p(j) * q(i-j) for j in range(self.deg + 1)]) for i in range(n)])

    def get(self, l):
        def f(i):
            if i >= 0:
                try:
                    return l[i]
                except:
                    return 0
            else:
                return 0
        return f

    def rou(self, n, k = 1):
        return pow(exp(complex(0, 2 * pi / n)), k)

    def even(self):
        return Polynomial([self.coeffs[2 * i] for i in range(self.deg // 2 + 1)])

    def odd(self):
        return Polynomial([self.coeffs[2 * i + 1] for i in range(self.deg // 2 + int(self.deg % 2 != 0))])

    def padded(self, limit = None):
        limit = 0 if limit is None else limit
        size = 1
        n = self.deg + 1
        while size < n or size < limit:
            size = size * 2
        return Polynomial(self.coeffs + (size - n) * [0])

    def cround(self, x, d):
        return complex(round(x.real, d), round(x.imag, d))

    def DFT(self, n = None, inv = False, dec = 2):
        Q = self.padded(n)
        n = Q.deg + 1
        def go(P, k):
            if P.deg == 0:
                return pow(1 / n, int(inv)) * P.coeffs[0] * np.ones(n)
            else:
                (evenDFT, oddDFT) = (go(P.even(), 2 * k), go(P.odd(), 2 * k))
                W = np.vectorize(lambda i: self.rou(n, k * i))(np.array(range(n)))
                return np.vectorize(lambda v: self.cround(v, dec))(evenDFT + W * oddDFT)
        return go(Q, pow(-1, int(inv)))

    def fast_mult(self, Q, dec = 2):
        n = self.deg + Q.deg + 1
        prod = self.DFT(n, dec = dec) * Q.DFT(n, dec = dec)
        coeffs = Polynomial(list(prod)).DFT(inv = True, dec = dec)
        return Polynomial([round(c.real, 0) for c in coeffs])

    def mytime(self, function, args, trials = None):
        def run():
            start = time.time()
            function(*args)
            return time.time() - start
        return run() if trials is None else sum([run() for t in range(trials)]) / trials

    def randomPoly(self, n):
        return Polynomial([random.randint(-9, 9) for i in range(n)])

    def checkCorrect(self, size):
        df = pd.DataFrame(index = range(size), columns=["P", "Q", "P * Q (n^2)", "P * Q (DFT)"])
        for i in df.index:
            (P, Q) = (self.randomPoly(i + 1), self.randomPoly(i + 1))
            df.loc[i, :] = [P, Q, P * Q, P.fast_mult(Q, dec=5)]
        df.to_csv("checkCorrect.csv")

    def timeAnalysis(self, size):
        start = time.time()
        df = pd.DataFrame(index=range(size // 50), columns=["n", "DFT Time (s)"])
        for i in df.index:
            print("-----------------------------------")
            n = 50 * (i + 1)
            print("n: {}".format(n))
            (P, Q) = (self.randomPoly(n), self.randomPoly(n))
            t = self.mytime(P.fast_mult, (Q, 5), 5)
            print("t: {} Seconds".format(t))
            df.loc[i, :] = [n, t]
        df.to_csv("timeAnalysis.csv")
        print("Total Time: {} Seconds".format(time.time() - start))






