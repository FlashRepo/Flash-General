from sk import rdivDemo
import pickle
import os
import numpy as np

folder = './PickleLocker/'
files = [folder + f for f in ['FlashB_25_110.p', 'NSGAII.p', 'SPEA2.p', 'moead.p', 'SWAY.p']]
problems = sorted(['P2', 'P3', 'P1', 'P4', 'M4', 'M1', 'M3', 'M2', 'X2', 'X3', 'X1', 'X4', 'X5'])

print files

measure = 'igd'
for problem in problems:
    print  problem
    dic = pickle.load(open('./PickleLocker/FlashB_25_110.p'))
    fl = ['Flash'] + [f[0] for f in dic[problem][measure]]

    dic = pickle.load(open('./PickleLocker/NSGAII.p'))
    n = ['NSGAII'] + [f[0] for f in dic[problem][measure]]

    dic = pickle.load(open('./PickleLocker/SPEA2.p'))
    s = ['SPEA2'] + [f[0] for f in dic[problem][measure]]

    dic = pickle.load(open('./PickleLocker/moead.p'))
    m = ['MOEAD'] + [f[0] for f in dic[problem][measure]]

    dic = pickle.load(open('./PickleLocker/SWAY.p'))
    sw = ['SWAY'] + [f[0] for f in dic[problem][measure]]

    rdivDemo(problem, [fl, n, s, m, sw], globalMinMax=False, isLatex=False)

