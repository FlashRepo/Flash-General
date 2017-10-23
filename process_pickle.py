import pickle
import os

pickle_folders = [name  for name in os.listdir(".") if os.path.isdir(name) and  'PickleLocker' in name]

files = ['M1', 'M2', 'M3', 'M4', 'P1', 'P2', 'P3', 'P4', 'X1', 'X2', 'X3', 'X4', 'X5', ]

for pickle_folder in pickle_folders:
    pickle_files = [pickle_folder + f for f in os.listdir(pickle_folder)]

    dict = {}
    for file in files:
        pfiles = [ pfile for pfile in pickle_files if file in pfile]
        print file, len(pfiles)
        for i,pfile in enumerate(pfiles):
            t = pickle.load(open(pfile))
            if i == 0:
                key = t.keys()[-1]
                dict[key] = {}
                dict[key]['evals'] = [t[key]['evals']]
                dict[key]['igd'] = [t[key]['igd']]
                dict[key]['gen_dist'] = [t[key]['gen_dist']]
                dict[key]['time'] = [t[key]['time']]

            dict[key]['evals'].append(t[key]['evals'])
            dict[key]['igd'].append(t[key]['igd'])
            dict[key]['gen_dist'].append(t[key]['gen_dist'])
            dict[key]['time'].append(t[key]['time'])

    pickle.dump(dict, open(pickle_folder + pickle_folder.replace('PickleLocker_', '') + '.p', 'w'))
