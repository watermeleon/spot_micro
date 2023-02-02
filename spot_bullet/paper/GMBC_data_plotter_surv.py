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
parser.add_argument("-nep","--NumberOfEpisodes", type=int, default=1000, 
                     help="Number of Episodes to Plot Data For")
parser.add_argument("-maw", "--MovingAverageWindow",  type=int, default=50,
                    help="Moving Average Window for Plotting (Default: 50)")
parser.add_argument("-surv","--Survival",
                    help="Plot Survival Curve", action='store_true')
parser.add_argument("-tr", "--TrainingData",
                    help="Plot Training Curve", action='store_true')
parser.add_argument("-tot", "--TotalReward",
                    help="Show Total Reward instead of Reward Per Timestep", action='store_true')
parser.add_argument("-ar","--RandAgentNum",
                    help="Randomized Agent Number To Load")
parser.add_argument("-anor", "--NoRandAgentNum",
                    help="Non-Randomized Agent Number To Load")
parser.add_argument("-raw", "--Raw",
                    help="Plot Raw Data in addition to Moving Averaged Data",action='store_true')
parser.add_argument("-s", "--Seed",  type=int, default=0, 
                    help="Seed [UP TO, e.g. 0 | 0, 1 | 0, 1, 2 ...] (Default: 0).")
parser.add_argument("-pout", "--PolicyOut",
                    help="Plot Policy Output Data", action='store_true')
parser.add_argument("-rough", "--Rough",
                    help="Plot Policy Output Data for Rough Terrain", action='store_true')
parser.add_argument("-tru", "--TrueAct",
                    help="Plot the Agent Action instead of what the robot sees", action='store_true')
parser.add_argument("-path1", "--path1")

ARGS = parser.parse_args()

MA_WINDOW = ARGS.MovingAverageWindow

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

    seed = ARGS.Seed
    nep = ARGS.NumberOfEpisodes
    training = ARGS.TrainingData
    surv = ARGS.Survival

    # Find abs path to this file
    my_path = os.path.abspath(os.path.dirname(__file__))
    results_path = os.path.join(my_path, "../results")

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    vanilla_surv = np.random.randn(1000)
    agent_surv = np.random.randn(1000)

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

    if surv:
        # Vanilla Data
        # if os.path.exists(results_path + "/" + file_name + "vanilla" +
        results_path2 = "../../../results/"

                        #   '_survival_{}'.format(nep)):
        with open(results_path2 + ARGS.path1, 'rb') as filehandle:
            vanilla_surv = np.array(pickle.load(filehandle))    
        print(vanilla_surv)    
        # # Vanilla Data
        # if os.path.exists(results_path + "/" + file_name + "vanilla" +
        #                   '_survival_{}'.format(nep)):
        #     with open(
        #             results_path + "/" + file_name + "vanilla" +
        #             '_survival_{}'.format(nep), 'rb') as filehandle:
        #         vanilla_surv = np.array(pickle.load(filehandle))

        # Rand Agent Data
        if os.path.exists(results_path + "/" + file_name +
                          "agent_{}".format(rand_agt) +
                          '_survival_{}'.format(nep)):
            with open(
                    results_path + "/" + file_name +
                    "agent_{}".format(rand_agt) + '_survival_{}'.format(nep),
                    'rb') as filehandle:
                d2gmbc_surv = np.array(pickle.load(filehandle))

        # NoRand Agent Data
        if os.path.exists(results_path + "/" + file_name +
                          "agent_{}".format(norand_agt) +
                          '_survival_{}'.format(nep)):
            with open(
                    results_path + "/" + file_name +
                    "agent_{}".format(norand_agt) + '_survival_{}'.format(nep),
                    'rb') as filehandle:
                gmbc_surv = np.array(pickle.load(filehandle))
                # print(gmbc_surv[:, 0])
        print(d2gmbc_surv)
        # Extract useful values
        vanilla_surv_x = vanilla_surv[:1000, 0]
        d2gmbc_surv_x = d2gmbc_surv[:, 0]
        gmbc_surv_x = gmbc_surv[:, 0]
        # convert the lists to series
        data = {
            'Open Loop': vanilla_surv_x,
            'GMBC': d2gmbc_surv_x,
            'D^2-GMBC': gmbc_surv_x
        }

        colors = ['r', 'g', 'b']

        # get dataframe
        df = pd.DataFrame(data)
        print(df)

        # get dataframe2
        # Extract useful values
        vanilla_surv_dt = vanilla_surv[:1000, -1]
        d2gmbc_surv_dt = d2gmbc_surv[:, -1]
        gmbc_surv_dt = gmbc_surv[:, -1]
        # convert the lists to series
        data2 = {
            'Open Loop': vanilla_surv_dt,
            'GMBC': d2gmbc_surv_dt,
            'D^2-GMBC': gmbc_surv_dt
        }
        df2 = pd.DataFrame(data2)

        # Plot
        for i, col in enumerate(df.columns):
            sns.distplot(df[[col]], color=colors[i])

        plt.legend(labels=['D^2-GMBC', 'GMBC', 'Open Loop'])
        plt.xlabel("Forward Survived Distance (m)")
        plt.ylabel("Kernel Density Estimate")
        plt.show()

        # Print AVG and STDEV
        norand_avg = np.average(copy.deepcopy(gmbc_surv_x))
        norand_std = np.std(copy.deepcopy(gmbc_surv_x))
        rand_avg = np.average(copy.deepcopy(d2gmbc_surv_x))
        rand_std = np.std(copy.deepcopy(d2gmbc_surv_x))
        vanilla_avg = np.average(copy.deepcopy(vanilla_surv_x))
        vanilla_std = np.std(copy.deepcopy(vanilla_surv_x))

        print("Open Loop: AVG [{}] | STD [{}] | AMOUNT [{}]".format(
            vanilla_avg, vanilla_std, gmbc_surv_x.shape[0]))
        print("D^2-GMBC: AVG [{}] | STD [{}] AMOUNT [{}]".format(
            rand_avg, rand_std, d2gmbc_surv_x.shape[0]))
        print("GMBC: AVG [{}] | STD [{}] AMOUNT [{}]".format(
            norand_avg, norand_std, vanilla_surv_x.shape[0]))

        # collect data
        gmbc_surv_x_less_5, gmbc_surv_num_less_5 = extract_data_bounds(
            0, 5, gmbc_surv_x, gmbc_surv_dt)
        d2gmbc_surv_x_less_5, d2gmbc_surv_num_less_5 = extract_data_bounds(
            0, 5, d2gmbc_surv_x, d2gmbc_surv_dt)
        vanilla_surv_x_less_5, vanilla_surv_num_less_5 = extract_data_bounds(
            0, 5, vanilla_surv_x, vanilla_surv_dt)

        # <=5
        # Make sure all arrays filled
        if gmbc_surv_x_less_5.size == 0:
            gmbc_surv_x_less_5 = np.array([0])
        if d2gmbc_surv_x_less_5.size == 0:
            d2gmbc_surv_x_less_5 = np.array([0])
        if vanilla_surv_x_less_5.size == 0:
            vanilla_surv_x_less_5 = np.array([0])

        norand_avg = np.average(gmbc_surv_x_less_5)
        norand_std = np.std(gmbc_surv_x_less_5)
        rand_avg = np.average(d2gmbc_surv_x_less_5)
        rand_std = np.std(d2gmbc_surv_x_less_5)
        vanilla_avg = np.average(vanilla_surv_x_less_5)
        vanilla_std = np.std(vanilla_surv_x_less_5)
        print("<= 5m")
        print(
            "Open Loop: AVG [{}] | STD [{}] | AMOUNT DEAD [{}] | AMOUNT ALIVE [{}]"
            .format(vanilla_avg, vanilla_std,
                    vanilla_surv_x_less_5.shape[0] - vanilla_surv_num_less_5,
                    vanilla_surv_num_less_5))
        print(
            "D^2-GMBC: AVG [{}] | STD [{}] AMOUNT DEAD [{}] | AMOUNT ALIVE [{}]"
            .format(rand_avg, rand_std,
                    d2gmbc_surv_x_less_5.shape[0] - d2gmbc_surv_num_less_5,
                    d2gmbc_surv_num_less_5))
        print("GMBC: AVG [{}] | STD [{}] AMOUNT DEAD [{}] | AMOUNT ALIVE [{}]".
              format(norand_avg, norand_std,
                     gmbc_surv_x_less_5.shape[0] - gmbc_surv_num_less_5,
                     gmbc_surv_num_less_5))

        # collect data
        gmbc_surv_x_gtr_5, gmbc_surv_num_gtr_5 = extract_data_bounds(
            5, 90, gmbc_surv_x, gmbc_surv_dt)
        d2gmbc_surv_x_gtr_5, d2gmbc_surv_num_gtr_5 = extract_data_bounds(
            5, 90, d2gmbc_surv_x, d2gmbc_surv_dt)
        vanilla_surv_x_gtr_5, vanilla_surv_num_gtr_5 = extract_data_bounds(
            5, 90, vanilla_surv_x, vanilla_surv_dt)

        # >5 <90
        # Make sure all arrays filled
        if gmbc_surv_x_gtr_5.size == 0:
            gmbc_surv_x_gtr_5 = np.array([0])
        if d2gmbc_surv_x_gtr_5.size == 0:
            d2gmbc_surv_x_gtr_5 = np.array([0])
        if vanilla_surv_x_gtr_5.size == 0:
            vanilla_surv_x_gtr_5 = np.array([0])

        norand_avg = np.average(gmbc_surv_x_gtr_5)
        norand_std = np.std(gmbc_surv_x_gtr_5)
        rand_avg = np.average(d2gmbc_surv_x_gtr_5)
        rand_std = np.std(d2gmbc_surv_x_gtr_5)
        vanilla_avg = np.average(vanilla_surv_x_gtr_5)
        vanilla_std = np.std(vanilla_surv_x_gtr_5)
        print("> 5m and <90m")
        print(
            "Open Loop: AVG [{}] | STD [{}] | AMOUNT DEAD [{}] | AMOUNT ALIVE [{}]"
            .format(vanilla_avg, vanilla_std,
                    vanilla_surv_x_gtr_5.shape[0] - vanilla_surv_num_gtr_5,
                    vanilla_surv_num_gtr_5))
        print(
            "D^2-GMBC: AVG [{}] | STD [{}] AMOUNT DEAD [{}] | AMOUNT ALIVE [{}]"
            .format(rand_avg, rand_std,
                    d2gmbc_surv_x_gtr_5.shape[0] - d2gmbc_surv_num_gtr_5,
                    d2gmbc_surv_num_gtr_5))
        print("GMBC: AVG [{}] | STD [{}] AMOUNT DEAD [{}] | AMOUNT ALIVE [{}]".
              format(norand_avg, norand_std,
                     gmbc_surv_x_gtr_5.shape[0] - gmbc_surv_num_gtr_5,
                     gmbc_surv_num_gtr_5))

        # collect data
        gmbc_surv_x_gtr_90, gmbc_surv_num_gtr_90 = extract_data_bounds(
            90, np.inf, gmbc_surv_x, gmbc_surv_dt)
        d2gmbc_surv_x_gtr_90, d2gmbc_surv_num_gtr_90 = extract_data_bounds(
            90, np.inf, d2gmbc_surv_x, d2gmbc_surv_dt)
        vanilla_surv_x_gtr_90, vanilla_surv_num_gtr_90 = extract_data_bounds(
            90, np.inf, vanilla_surv_x, vanilla_surv_dt)

        # >90
        # Make sure all arrays filled
        if gmbc_surv_x_gtr_90.size == 0:
            gmbc_surv_x_gtr_90 = np.array([0])
        if d2gmbc_surv_x_gtr_90.size == 0:
            d2gmbc_surv_x_gtr_90 = np.array([0])
        if vanilla_surv_x_gtr_90.size == 0:
            vanilla_surv_x_gtr_90 = np.array([0])

        norand_avg = np.average(gmbc_surv_x_gtr_90)
        norand_std = np.std(gmbc_surv_x_gtr_90)
        rand_avg = np.average(d2gmbc_surv_x_gtr_90)
        rand_std = np.std(d2gmbc_surv_x_gtr_90)
        vanilla_avg = np.average(vanilla_surv_x_gtr_90)
        vanilla_std = np.std(vanilla_surv_x_gtr_90)
        print(">= 90m")
        print(
            "Open Loop: AVG [{}] | STD [{}] | AAMOUNT DEAD [{}] | AMOUNT ALIVE [{}]"
            .format(vanilla_avg, vanilla_std,
                    vanilla_surv_x_gtr_90.shape[0] - vanilla_surv_num_gtr_90,
                    vanilla_surv_num_gtr_90))
        print(
            "D^2-GMBC: AVG [{}] | STD [{}] AMOUNT DEAD [{}] | AMOUNT ALIVE [{}]"
            .format(rand_avg, rand_std,
                    d2gmbc_surv_x_gtr_90.shape[0] - d2gmbc_surv_num_gtr_90,
                    d2gmbc_surv_num_gtr_90))
        print("GMBC: AVG [{}] | STD [{}] AMOUNT DEAD [{}] | AMOUNT ALIVE [{}]".
              format(norand_avg, norand_std,
                     gmbc_surv_x_gtr_90.shape[0] - gmbc_surv_num_gtr_90,
                     gmbc_surv_num_gtr_90))

        # Save to excel
        df.to_excel(results_path + "/SurvDist.xlsx", index=False)
        df2.to_excel(results_path + "/SurvDT.xlsx", index=False)

   

if __name__ == '__main__':
    main()
