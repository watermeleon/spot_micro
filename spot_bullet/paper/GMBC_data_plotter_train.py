#!/usr/bin/env python

import numpy as np
import sys
import os
import argparse
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import copy
from scipy.stats import norm
sns.set()

# ARGUMENTS
descr = "Spot Mini Mini ARS Agent Evaluator."
parser = argparse.ArgumentParser(description=descr)
parser.add_argument("-nep",
                    "--NumberOfEpisodes",
                    help="Number of Episodes to Plot Data For")

parser.add_argument("-path1", "--path1")
parser.add_argument("-path2","--path2")
parser.add_argument("-maw",
                    "--MovingAverageWindow",
                    help="Moving Average Window for Plotting (Default: 50)")
parser.add_argument("-surv",
                    "--Survival",
                    help="Plot Survival Curve",
                    action='store_true')
parser.add_argument("-tr",
                    "--TrainingData",
                    help="Plot Training Curve",
                    action='store_true')
parser.add_argument("-tot",
                    "--TotalReward",
                    help="Show Total Reward instead of Reward Per Timestep",
                    action='store_true')
parser.add_argument("-ar",
                    "--RandAgentNum",
                    help="Randomized Agent Number To Load")
parser.add_argument("-anor",
                    "--NoRandAgentNum",
                    help="Non-Randomized Agent Number To Load")
parser.add_argument("-raw",
                    "--Raw",
                    help="Plot Raw Data in addition to Moving Averaged Data",
                    action='store_true')
parser.add_argument(
    "-s",
    "--Seed",
    help="Seed [UP TO, e.g. 0 | 0, 1 | 0, 1, 2 ...] (Default: 0).")
parser.add_argument("-pout",
                    "--PolicyOut",
                    help="Plot Policy Output Data",
                    action='store_true')
parser.add_argument("-rough",
                    "--Rough",
                    help="Plot Policy Output Data for Rough Terrain",
                    action='store_true')
parser.add_argument(
    "-tru",
    "--TrueAct",
    help="Plot the Agent Action instead of what the robot sees",
    action='store_true')
ARGS = parser.parse_args()
ARGS.TrainingData = True

MA_WINDOW = 50
if ARGS.MovingAverageWindow:
    MA_WINDOW = int(ARGS.MovingAverageWindow)


def moving_average(a, n=MA_WINDOW):
    MA = np.cumsum(a, dtype=float)
    MA[n:] = MA[n:] - MA[:-n]
    return MA[n - 1:] / n


def extract_data_bounds(min=0, max=5, dist_data=None, dt_data=None):
    """ 3 bounds: lower, mid, highest
    """

    if dist_data is not None:

        # Get Survival Data, dt
        # Lowest Bound: x <= max
        bound = np.array([0])
        if min == 0:
            less_max_cond = dist_data <= max
            bound = np.where(less_max_cond)
        else:
            # Highest Bound: min <= x
            if max == np.inf:
                gtr_min_cond = dist_data >= min
                bound = np.where(gtr_min_cond)
            # Mid Bound: min < x < max
            else:
                less_max_gtr_min_cond = np.logical_and(dist_data > min,
                                                       dist_data < max)
                bound = np.where(less_max_gtr_min_cond)

        if dt_data is not None:
            dt_bounded = dt_data[bound]
            num_surv = np.array(np.where(dt_bounded == 50000))[0].shape[0]
        else:
            num_surv = None

        return dist_data[bound], num_surv
    else:
        return None


def main():
    """ The main() function. """
    file_name = "spot_ars_"

    seed = 1
    if ARGS.Seed:
        seed = ARGS.Seed

    # Find abs path to this file
    my_path = os.path.abspath(os.path.dirname(__file__))
    results_path = os.path.join(my_path, "../results")

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    vanilla_surv = np.random.randn(1000)
    agent_surv = np.random.randn(1000)

    nep = 1000

    if ARGS.NumberOfEpisodes:
        nep = ARGS.NumberOfEpisodes
    if ARGS.TrainingData:
        training = True
    else:
        training = False
    if ARGS.Survival:
        surv = True
    else:
        surv = False
    if ARGS.PolicyOut or ARGS.Rough or ARGS.TrueAct:
        pout = True
    else:
        pout = False

    if not pout and not surv and not training:
        print(
            "Please Select which Data you would like to plot (-pout | -surv | -tr)"
        )
    rand_agt = 579
    norand_agt = 569
    if ARGS.RandAgentNum:
        rand_agt = ARGS.RandAgentNum
    if ARGS.NoRandAgentNum:
        norand_agt = ARGS.NoRandAgentNum

    print("leon is here", ARGS.TrainingData, training)
    if training:
        rand_data_list = []
        norand_data_list = []
        rand_shortest_length = np.inf
        norand_shortest_length = np.inf
        for i in range(int(seed) ):
            # Training Data Plotter
            # rand_data_temp = np.load(results_path + "/spot_ars_rand_" +
            #                          "seed" + str(i) + ".npy")
            # norand_data_temp = np.load(results_path + "/spot_ars_norand_" +
            #                            "seed" + str(i) + ".npy")
            # rand_data_temp = np.load(results_path2 + "spot_ars_rand_seed0.npy")
            # norand_data_temp = np.load(results_path2 + "spot_ars_norand_seed0.npy")

            results_path2 = "../../../results/final5_5july/"
            print("path:",results_path2 + ARGS.path1+"_seed"+str(i) + ".npy" )

            rand_data_temp = np.load(results_path2 + ARGS.path1+"_seed"+str(i) + ".npy")
            print("file data:", rand_data_temp)

            norand_data_temp = np.load(results_path2 + ARGS.path2+"_seed"+str(i) + ".npy")
            rand_shortest_length = min(
                np.shape(rand_data_temp[:, 1])[0], rand_shortest_length)
            norand_shortest_length = min(
                np.shape(norand_data_temp[:, 1])[0], norand_shortest_length)

            rand_data_list.append(rand_data_temp)
            norand_data_list.append(norand_data_temp)

        tot_rand_data = []
        tot_norand_data = []
        norm_rand_data = []
        norm_norand_data = []
        for i in range(int(seed)) :
            tot_rand_data.append(
                moving_average(rand_data_list[i][:rand_shortest_length, 0]))
            tot_norand_data.append(
                moving_average(
                    norand_data_list[i][:norand_shortest_length, 0]))
            norm_rand_data.append(
                moving_average(rand_data_list[i][:rand_shortest_length, 1]))
            norm_norand_data.append(
                moving_average(
                    norand_data_list[i][:norand_shortest_length, 1]))

        tot_rand_data = np.array(tot_rand_data)
        tot_norand_data = np.array(tot_norand_data)
        norm_rand_data = np.array(norm_rand_data)
        norm_norand_data = np.array(norm_norand_data)

        # column-wise
        axis = 0

        # MEAN
        tot_rand_mean = tot_rand_data.mean(axis=axis)
        tot_norand_mean = tot_norand_data.mean(axis=axis)
        norm_rand_mean = norm_rand_data.mean(axis=axis)
        norm_norand_mean = norm_norand_data.mean(axis=axis)

        # STD
        tot_rand_std = tot_rand_data.std(axis=axis)
        tot_norand_std = tot_norand_data.std(axis=axis)
        norm_rand_std = norm_rand_data.std(axis=axis)
        norm_norand_std = norm_norand_data.std(axis=axis)

        aranged_rand = np.arange(np.shape(tot_rand_mean)[0])
        aranged_norand = np.arange(np.shape(tot_norand_mean)[0])

        if ARGS.TotalReward:
            if ARGS.Raw:
                plt.plot(rand_data_list[0][:, 0],
                         label="SAC  (Total Reward)",
                         color='g')
                plt.plot(norand_data_list[0][:, 0],
                         label="Non-Randomized (Total Reward)",
                         color='r')
            plt.plot(aranged_norand,
                     tot_norand_mean,
                     label="MA: Non-Randomized (Total Reward)",
                     color='r')
            plt.fill_between(aranged_norand,
                             tot_norand_mean - tot_norand_std,
                             tot_norand_mean + tot_norand_std,
                             color='r',
                             alpha=0.2)
            plt.plot(aranged_rand,
                     tot_rand_mean,
                     label="MA: Randomized (Total Reward)",
                     color='g')
            plt.fill_between(aranged_rand,
                             tot_rand_mean - tot_rand_std,
                             tot_rand_mean + tot_rand_std,
                             color='g',
                             alpha=0.2)
        else:
            if ARGS.Raw:
                plt.plot(rand_data_list[0][:, 1],
                         label="Randomized (Reward/dt)",
                         color='g')
                plt.plot(norand_data_list[0][:, 1],
                         label="Non-Randomized (Reward/dt)",
                         color='r')
            # plt.plot(aranged_norand,
            #          norm_norand_mean,
            #          label="MA: Non-Randomized (Reward/dt)",
            #          color='r')
            plt.fill_between(aranged_norand,
                             norm_norand_mean - norm_norand_std,
                             norm_norand_mean + norm_norand_std,
                             color='r',
                             alpha=0.2)
            plt.plot(aranged_rand,
                     norm_rand_mean,
                     label="MA: ARS (Reward/dt)",
                     color='r')
            plt.fill_between(aranged_rand,
                             norm_rand_mean - norm_rand_std,
                             norm_rand_mean + norm_rand_std,
                             color='r',
                             alpha=0.2)
        plt.xlabel("Epoch #")
        plt.ylabel("Reward")
        plt.title(
            "Simulation Training Performance with {} seed samples".format(int(seed) ))
        plt.legend(loc='lower right', title="RL agent:")
        plt.show()




if __name__ == '__main__':
    main()
