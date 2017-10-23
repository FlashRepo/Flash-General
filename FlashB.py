from __future__ import division
import numpy as np
import os
import sys
from random import shuffle
import time
sys.path.append("/Users/viveknair/GIT/Flash-MultiConfig/")
from utility import  generational_distance, inverted_generational_distance, read_file
from non_dominated_sort import non_dominated_sort, binary_domination


lessismore = {}
lessismore["M1"] = [True, True, True]
lessismore["M2"] = [True, True, True]
lessismore["M3"] = [True, True, True]
lessismore["M4"] = [True, True, True]

lessismore["X1"] = [True, True, True, True]
lessismore["X2"] = [True, True, True, True]
lessismore["X3"] = [True, True, True, True]
lessismore["X4"] = [True, True, True, True]
lessismore["X5"] = [True, True, True, True]

lessismore["P1"] = [True, True, True]
lessismore["P2"] = [True, True, True]
lessismore["P3"] = [True, True, True]
lessismore["P4"] = [True, True, True]




ranges = {}
ranges["M1"] = [[95394.0, 96983.0], [400.0, 605.0], [99341.0, 99574.0]]
ranges["M2"] = [[94994.0, 97219.0], [452.0, 727.0], [99466.0, 99660.0]]
ranges["M3"] = [[94759.0, 96679.0], [417.0, 642.0], [99383.0, 99595.0]]
ranges["M4"] = [[92403.0, 95344.0], [428.0, 696.0], [99370.0, 99627.0]]

ranges["X1"] = [[5.8900921014900005, 28583.461233399998], [5.70862368202, 98.79220126530001], [14.9038336217, 791879.990629], [0.0, 14.745308310999999]]
ranges["X2"] = [[5.07704875571, 23004.2641148], [5.79962055412, 98.2239438536], [10.6753616341, 428117.623585], [0.0, 13.941018766800001]]
ranges["X3"] = [[4.784674809519999, 27522.1840857], [4.2245581331699995, 102.89673937799999], [19.1702767897, 372508.726334], [0.0, 13.941018766800001]]
ranges["X4"] = [[4.36140164564, 28090.846327799998], [5.13691252687, 114.196144121], [5.541133544419999, 401407.569903], [0.0, 13.1367292225]]
ranges["X5"] = [[4.48249903236, 22162.0418187], [5.76103797422, 103.62850582200001], [8.09558500714, 312806.337078], [0.0, 14.745308310999999]]

ranges["P1"] = [[50.41390895399999, 2884.36190927], [-2.22044604925e-16, 0.841889480617], [0.0, 0.7539882451719999]]
ranges["P2"] = [[0.0, 34227.640271599994], [0.0, 1.0], [0.0, 0.827586206897]]
ranges["P3"] = [[202.22098459400002, 2776.06783571], [0.36918150500299995, 0.7269238731450001], [0.0, 0.699346405229]]
ranges["P4"] = [[0.0, 1459.07484037], [-2.22044604925e-16, 1.0], [0.0, 0.7272727272730001]]






def get_nd_solutions(filename, train_indep, training_dep, testing_indep):
    no_of_objectives = len(training_dep[0])
    predicted_objectives = []
    for objective_no in xrange(no_of_objectives):
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor()
        model.fit(train_indep, [t[objective_no] for t in training_dep])
        predicted = model.predict(testing_indep)
        predicted_objectives.append(predicted)

    # Merge the objectives
    merged_predicted_objectves = []
    for i in xrange(len(predicted_objectives[0])):
        merged_predicted_objectves.append([predicted_objectives[obj_no][i] for obj_no in xrange(no_of_objectives)])
    assert(len(merged_predicted_objectves) == len(testing_indep)), "Something is wrong"

    # Find Non-Dominated Solutions
    pf_indexes = non_dominated_sort(merged_predicted_objectves, lessismore[filename], [r[0] for r in ranges], [r[1] for r in ranges])
    # print "Number of ND Solutions: ", len(pf_indexes)

    return [testing_indep[i] for i in pf_indexes], [merged_predicted_objectves[i] for i in pf_indexes]

def normalize(x, min, max):
    tmp = float((x - min)) / (max - min + 0.000001)
    if tmp > 1: return 1
    elif tmp < 0: return 0
    else: return tmp


def get_next_points(file, training_indep, training_dep, testing_indep, directions):
    no_of_objectives = len(training_dep[0])

    predicted_objectives = []
    for objective_no in xrange(no_of_objectives):
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor()
        model.fit(training_indep, [t[objective_no] for t in training_dep])
        predicted = model.predict(testing_indep)
        predicted_objectives.append(predicted)

    # Merge the objectives
    merged_predicted_objectves = []
    for i in xrange(len(predicted_objectives[0])):
        merged_predicted_objectves.append([predicted_objectives[obj_no][i] for obj_no in xrange(no_of_objectives)])
    assert (len(merged_predicted_objectves) == len(testing_indep)), "Something is wrong"


    # Convert the merged_predicted_objectives to minimization problem
    lism = lessismore[file]
    dependents = []
    for rd in merged_predicted_objectves:
        temp = []
        for i in xrange(len(lism)):
            # if lessismore[i] is true - Minimization else Maximization
            if lism[i] is False:
                temp.append(-1 * rd[i])
            else:
                temp.append(rd[i])
        dependents.append(temp)

    # Normalize objectives
    mins = [r[0] for r in ranges[file]]
    maxs = [r[1] for r in ranges[file]]

    normalized_dependents = []
    for dependent in dependents:
        normalized_dependents.append([normalize(dependent[i], mins[i], maxs[i]) for i in xrange(no_of_objectives)])
    assert(len(normalized_dependents) == len(dependents)), "Something is wrong"

    return_indexes = []
    for direction in directions:
        transformed = []
        for dependent in normalized_dependents:
            assert(len(direction) == len(dependent)), "Something is wrong"
            transformed.append(sum([i*j for i, j in zip(direction, dependent)]))
        return_indexes.append(transformed.index(min(transformed)))
    assert(len(return_indexes) == len(directions)), "Something is wrong"

    return_indexes = list(set(return_indexes))
    return return_indexes

def get_random_numbers(len_of_objectives):
    from random import random
    random_numbers = [random() for _ in xrange(len_of_objectives)]
    ret = [num/sum(random_numbers) for num in random_numbers]
    # print ret, sum(ret), int(sum(ret))==1
    # assert(int(sum(ret)) == 1), "Something is wrong"
    return ret


def run_main(files, repeat_no, stop, start_size):
    initial_time = time.time()
    all_data = {}
    initial_sample_size = start_size
    for file in files:
        all_data[file] = {}
        all_data[file]['evals'] = []
        all_data[file]['gen_dist'] = []
        all_data[file]['igd'] = []

        print file
        data = read_file('./Data/' + file + '.csv')

        # Creating Objective Dict
        objectives_dict = {}
        for d in data:
            key = ",".join(map(str, d.decisions))
            objectives_dict[key] = d.objectives

        number_of_objectives = len(data[0].objectives)
        number_of_directions = 10

        directions = [get_random_numbers(number_of_objectives) for _ in xrange(number_of_directions)]
        shuffle(data)

        training_indep = [d.decisions for d in data[:initial_sample_size]]
        testing_indep = [d.decisions for d in data[initial_sample_size:]]

        while True:
            print ". ",
            sys.stdout.flush()

            def get_objective_score(independent):
                key = ",".join(map(str, independent))
                return objectives_dict[key]

            training_dep = [get_objective_score(r) for r in training_indep]

            next_point_indexes = get_next_points(file, training_indep, training_dep, testing_indep, directions)
            # print "Points Sampled: ", next_point_indexes
            next_point_indexes = sorted(next_point_indexes, reverse=True)
            for next_point_index in next_point_indexes:
                temp = testing_indep[next_point_index]
                del testing_indep[next_point_index]
                training_indep.append(temp)
            # print len(training_indep), len(testing_indep), len(data)
            assert(len(training_indep) + len(testing_indep) == len(data)), "Something is wrong"
            if len(training_indep) > stop: break


        print
        print "Size of the frontier = ", len(training_indep), " Evals: ", len(training_indep),
        # Calculate the True ND
        training_dependent = [get_objective_score(r) for r in training_indep]
        approx_dependent_index = non_dominated_sort(training_dependent, lessismore[file], [r[0] for r in ranges[file]],
                                             [r[1] for r in ranges[file]])
        approx_dependent = sorted([training_dependent[i] for i in approx_dependent_index], key=lambda x: x[0])
        all_data[file]['evals'].append(len(training_indep))

        actual_dependent = [d.objectives for d in data]
        true_pf_indexes = non_dominated_sort(actual_dependent, lessismore[file], [r[0] for r in ranges[file]],
                                             [r[1] for r in ranges[file]])
        true_pf = sorted([actual_dependent[i] for i in true_pf_indexes], key=lambda x: x[0])
        print "Length of True PF: " , len(true_pf),
        print "Length of the Actual PF: ", len(training_dependent),
        all_data[file]['gen_dist'].append(generational_distance(true_pf, approx_dependent, ranges[file]))
        all_data[file]['igd'].append(inverted_generational_distance(true_pf, approx_dependent, ranges[file]))

        print " GD: ", all_data[file]['gen_dist'][-1],
        print " IGD: ", all_data[file]['igd'][-1]
        all_data[file]['time'] = time.time() - initial_time
        # print all_data[file]['time']
        try:
            os.mkdir('PickleLocker_FlashB_'+str(start_size)+'_'+str(stop))
        except: pass

        import pickle
        pickle.dump(all_data, open('PickleLocker_FlashB_'+str(start_size)+'_'+str(stop)+'/' + file + '_' + str(repeat_no) + '.p', 'w'))

if __name__ == "__main__":
    files = ['M1', 'M2', 'M3', 'M4', 'P1', 'P2', 'P3', 'P4', 'X1', 'X2', 'X3', 'X4', 'X5', ]

    import multiprocessing as mp

    times = {}
    # Main control loop
    pool = mp.Pool()
    for file in files:
        times[file] = []
        for budget in [30, 50, 70, 90, 110]:
            for start_size in [15, 20, 25, 30]:
                for rep in xrange(20):
                    pool.apply_async(run_main, ([file], rep, budget, start_size))
                    # start_time = time()
                    # run_main([file], rep, 50, start_size)
                    # times[file].append(time() - start_time)

    pool.close()
    pool.join()
