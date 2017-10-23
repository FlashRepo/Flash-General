import pickle
import os

problems = ['POM', 'XOMO', 'monrp']

pickle_files = [f for f in os.listdir('.') if '.p' in f and '.py' not in f and 'SWAY5' in f]
print pickle_files
name_mapping = {
    'POM3A' : 'P1',
    'POM3B' : 'P2',
    'POM3C' : 'P3',
    'POM3D' : 'P4',
    'xomo_all' : 'X1',
    'xomo_flight' : 'X2',
    'xomo_ground' : 'X3',
    'xomo_osp' : 'X4',
    'xomoo2' : 'X5',
    'MONRP_50_4_5_0_90' : 'M1',
    'MONRP_50_4_5_0_110' : 'M2',
    'MONRP_50_4_5_4_90' : 'M3',
    'MONRP_50_4_5_4_110' : 'M4',
}

moead = {}
NSGAII = {}
SPEA2 = {}
SWAY5 = {}

for problem in problems:
    for pickle_file in pickle_files:
        if problem not in pickle_file: continue
        t = pickle.load(open(pickle_file))
        # print t
        for tt in t.keys():
            SWAY5[name_mapping[tt]] = {}
            SWAY5[name_mapping[tt]]['evals'] = [[ttt] for ttt in t[tt]['evals']]
            SWAY5[name_mapping[tt]]['igd'] = [[ttt] for ttt in t[tt]['igd']]
            SWAY5[name_mapping[tt]]['gen_dist'] = [[ttt] for ttt in t[tt]['gen_dist']]
    # print NSGAII.keys()

pickle.dump(SWAY5, open('SWAY5.p', 'w'))