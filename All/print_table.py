import pickle
import os
import numpy as np

pickle_files = [f for f in os.listdir('.') if '.py' not in f and '.p' in f]

problems = sorted(['P2', 'P3', 'P1', 'P4', 'M4', 'M1', 'M3', 'M2', 'X2', 'X3', 'X1', 'X4', 'X5'])
solutions = sorted(pickle_files)
freq_gd = {}
freq_igd = {}
for solution in solutions:
    freq_gd[solution] = 0
    freq_igd[solution] = 0

for problem in problems:
    al = []
    for solution in solutions:
        dict = pickle.load(open(solution))
        gd = dict[problem]['gen_dist']
        igd = dict[problem]['igd']
        evals = dict[problem]['evals']

        al.append([solution, round(np.median(gd), 3), round(np.median(igd), 3), np.median(evals)])
    freq_gd[sorted(al, key=lambda x:x[1])[0][0]] += 1
    freq_igd[sorted(al, key=lambda x:x[2])[0][0]] += 1

for key in freq_gd.keys(): print key, freq_gd[key]
print
for key in freq_igd.keys(): print key, freq_igd[key]

import pdb
pdb.set_trace()
