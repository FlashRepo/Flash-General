import pickle
import numpy as np

folder = './PickleLocker/'
files = [folder + f for f in ['FlashB_25_110.p', 'NSGAII.p', 'SPEA2.p', 'moead.p', 'SWAY.p']]
problems = sorted(['P2', 'P3', 'P1', 'P4', 'M4', 'M1', 'M3', 'M2', 'X2', 'X3', 'X1', 'X4', 'X5'])


for measure in ['evals', 'gen_dist', 'igd']:
    for problem in problems:
        print problem,
        for file in files:
            dic = pickle.load(open(file))
            # print dic.keys()
            try:
                print np.median([f[0] for f in dic[problem][measure]]),
            except:
                print np.median([f[0] for f in dic[problem][measure]]),
        print
    import pdb
    pdb.set_trace()