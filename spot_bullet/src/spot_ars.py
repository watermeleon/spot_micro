#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import sys

sys.path.append('../../')

from spotmicro.util.gui import GUI
from spotmicro.GymEnvs.spot_bezier_env import spotBezierEnv
from spotmicro.Kinematics.SpotKinematics import SpotModel
from spotmicro.GaitGenerator.Bezier import BezierGait
from spotmicro.OpenLoopSM.SpotOL import BezierStepper
from spotmicro.spot_env_randomizer import SpotEnvRandomizer

import time
from ars_lib.ars import ARSAgent, Normalizer, Policy, ParallelWorker
# Multiprocessing package for python
# Parallelization improvements based on:
# https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/ARS/ars.py
import multiprocessing as mp
from multiprocessing import Pipe

import os

import argparse

# ARGUMENTS
descr = "Spot Mini Mini ARS Agent Trainer."
parser = argparse.ArgumentParser(description=descr)
parser.add_argument("--agent_name", type=str, default="", help="Agent Number To Load")

parser.add_argument("-a", "--AgentNum", type=int, default=0, help="Agent Number To Load")
parser.add_argument("-hf", "--height_field",
                    help="Use height_field", action='store_true')
parser.add_argument("--load_agent", action='store_true')
parser.add_argument("-cs", "--contact_sensing",
                    help="Use Contact Sensing", action='store_true')
parser.add_argument("-dr", "--DontRandomize",
                    help="Do NOT Randomize State and Environment.", action='store_true')
parser.add_argument("-s", "--Seed",  type=int, default=0,  help="Seed (Default: 0).")
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument("-steps","--max_steps", type=int, default=5000,
                    help="Maximum number of Steps per episode")
# parser.add_argument('--agent_type', type=str, default="rand", choices=['rand', 'norand'])
parser.add_argument('--agent_type', type=str, default="rand", choices=['rand', 'norand',"dynamOnly"])
parser.add_argument('--distance_weight', type=int, default=1000)


ARGS = parser.parse_args()
print(ARGS)
print("Leons settings: never contact sense, both randomization changed/overwritten by: agent_type")


if ARGS.agent_type == "rand":
    ARGS.DontRandomize = False
    ARGS.height_field = True
elif ARGS.agent_type == "norand":
    ARGS.DontRandomize = True
    ARGS.height_field = False
elif ARGS.agent_type == "dynamOnly":
    print("using dynamics only")
    ARGS.DontRandomize = False
    ARGS.height_field = False

print("new args:", ARGS)
rand_name = ARGS.agent_type +"_"

# Messages for Pipe
_RESET = 1
_CLOSE = 2
_EXPLORE = 3


def main():
    """ The main() function. """
    # Hold mp pipes
    mp.freeze_support()
    print("STARTING SPOT TRAINING ENV")

    seed = ARGS.Seed
    print("SEED: {}".format(seed))
    max_timesteps = 4e7
    eval_freq = 10
    save_model = True
    file_name = "spot_ars_"
    if len(ARGS.agent_name) !=0:
        file_name += ARGS.agent_name + "_"


    if ARGS.DontRandomize:
        env_randomizer = None
    else:
        env_randomizer = SpotEnvRandomizer()

    # Find abs path to this file
    my_path = os.path.abspath(os.path.dirname(__file__))
    results_path = os.path.join(my_path, "results")
    if ARGS.contact_sensing:
        models_path = os.path.join(my_path, "models/contact")
    else:
        models_path = os.path.join(my_path, "models/no_contact")
    
    # models_path += "/ARS"
    # if not os.path.exists(results_path):
    #     os.makedirs(results_path)

    if not os.path.exists(models_path):
        os.makedirs(models_path)

    sacpath = models_path +"/ARS/" + file_name + rand_name[:-1] + "_s"+ str(seed)
    if not os.path.exists(sacpath):
        os.makedirs(sacpath)
    models_path = sacpath
    print("models path:", models_path)

    env = spotBezierEnv(render=False,
                            on_rack=False,
                            height_field=ARGS.height_field,
                            draw_foot_path=False,
                            contacts=ARGS.contact_sensing,
                            env_randomizer=env_randomizer,
                            distance_weight= ARGS.distance_weight,
                            mod_rew=True)

    # Set seeds
    env.seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    print("STATE DIM: {}".format(state_dim))
    action_dim = env.action_space.shape[0]
    print("ACTION DIM: {}".format(action_dim))
    max_action = float(env.action_space.high[0])

    env.reset()

    g_u_i = GUI(env.spot.quadruped)

    spot = SpotModel()
    T_bf = spot.WorldToFoot

    bz_step = BezierStepper(dt=env._time_step)
    bzg = BezierGait(dt=env._time_step)

    # Initialize Normalizer
    normalizer = Normalizer(state_dim)

    # Initialize Policy
    policy = Policy(state_dim, action_dim, seed=seed, episode_steps=ARGS.max_steps)

    # Initialize Agent with normalizer, policy and gym env
    agent = ARSAgent(normalizer, policy, env, bz_step, bzg, spot)
    agent_path = models_path + "/" + str(file_name) + rand_name + "seed" + str(seed) + "_"+ str(ARGS.AgentNum)
    results_path_full = results_path + "/" + str(file_name) + rand_name + "seed" + str(seed)

    episode_num = 0

    if ARGS.load_agent:
        if os.path.exists(agent_path + "_policy"):
            print("Loading Existing agent")
            agent.load(agent_path)
            agent.policy.episode_steps = ARGS.max_steps
            policy = agent.policy
        else:
            print("couldn't find agent for path:", agent_path)
            quit()
        res = np.load(results_path_full)
        episode_num = res.shape(0)


    env.reset(agent.desired_velocity, agent.desired_rate)

    # episode_reward = 0
    # episode_timesteps = 0
    # episode_num = 0

    # Create mp pipes
    num_processes = policy.num_deltas
    processes = []
    childPipes = []
    parentPipes = []

    # Store mp pipes
    for pr in range(num_processes):
        parentPipe, childPipe = Pipe()
        parentPipes.append(parentPipe)
        childPipes.append(childPipe)

    # Start multiprocessing
    # Start multiprocessing
    for proc_num in range(num_processes):
        p = mp.Process(target=ParallelWorker,
                       args=(childPipes[proc_num], env, state_dim))
        p.start()
        processes.append(p)

    print("STARTED SPOT TRAINING ENV")
    t = 0
    while t < (int(max_timesteps)):
        # Maximum timesteps per rollout
        episode_reward, episode_timesteps, reward_compare = agent.train_parallel(parentPipes)
        t += episode_timesteps

        # episode_reward = agent.train()
        # +1 to account for 0 indexing.
        # +0 on ep_timesteps since it will increment +1 even if done=True
        print(
            "Total T: {} Episode Num: {} Episode T: {} Reward: {:.2f} REWARD PER STEP: {:.2f} REW compare: {:.2f}"
            .format(t + 1, episode_num, episode_timesteps, episode_reward,
                    episode_reward / float(episode_timesteps), float(reward_compare)))

        # Store Results (concat)
        if episode_num == 0:
            res = np.array(
                [[episode_reward, episode_reward / float(episode_timesteps), reward_compare]])
        else:
            new_res = np.array(
                [[episode_reward, episode_reward / float(episode_timesteps), reward_compare]])
            res = np.concatenate((res, new_res))

        # Also Save Results So Far (Overwrite)
        # Results contain 2D numpy array of total reward for each ep
        # and reward per timestep for each ep
        np.save(results_path_full, res)

        # Evaluate episode
        if (episode_num + 1) % eval_freq == 0 or episode_num > ARGS.num_epochs:
            # print("saving")
            if save_model:
                agent.save(models_path + "/" + str(file_name) + rand_name + "seed" + 
                            str(seed) + "_"+ str(episode_num))
        # else:
        #     print("so not saving",(episode_num + 1) % eval_freq, episode_num, eval_freq )

        episode_num += 1

        if episode_num > ARGS.num_epochs:
            break
    # Close pipes and hence envs
    for parentPipe in parentPipes:
        parentPipe.send([_CLOSE, "pay2"])

    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
