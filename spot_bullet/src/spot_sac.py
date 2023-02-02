#!/usr/bin/env python
import sys
import numpy as np

sys.path.append('../../')

from spot_bullet.src.sac_lib import SoftActorCritic, NormalizedActions, ReplayBuffer, PolicyNetwork
import copy
from gym import spaces

import sys

sys.path.append('../')

from spotmicro.GymEnvs.spot_bezier_env import spotBezierEnv
from spotmicro.Kinematics.SpotKinematics import SpotModel
from spotmicro.GaitGenerator.Bezier import BezierGait
from spotmicro.spot_env_randomizer import SpotEnvRandomizer
from ars_lib.ars import Normalizer

# TESTING
from spotmicro.OpenLoopSM.SpotOL import BezierStepper

import time

import torch
import os
import argparse


CD_SCALE = 0.05
SLV_SCALE = 0.05
RESIDUALS_SCALE = 0.015
Z_SCALE = 0.05

# Filter actions
alpha = 0.7

# Added this to avoid filtering residuals
# -1 for all
actions_to_filter = 14

# For auto yaw control
P_yaw = 5.0

# ARGUMENTS
descr = "Spot Mini Mini ARS Agent Trainer."
parser = argparse.ArgumentParser(description=descr)
parser.add_argument("--agent_name", type=str, default="", help="Agent Number To Load")

parser.add_argument("-a", "--AgentNum", type=int, default=0, help="Agent Number To Load")
parser.add_argument("-hf", "--height_field",
                    help="Use height_field", action='store_true')
parser.add_argument("-cs", "--contact_sensing",
                    help="Use Contact Sensing", action='store_true')
parser.add_argument("--use_alpha20s", action='store_true')
parser.add_argument("-dr", "--DontRandomize",
                    help="Do NOT Randomize State and Environment.", action='store_true')
parser.add_argument("-s", "--Seed",  type=int, default=0,  help="Seed (Default: 0).")
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument("-steps","--max_steps", type=int, default=5000,
                    help="Maximum number of Steps per episode")
parser.add_argument('--agent_type', type=str, default="rand", choices=['rand', 'norand',"dynamOnly"])
parser.add_argument('--learningrate', type=float, default=3e-3)
parser.add_argument('--distance_weight', type=int, default=1000)
parser.add_argument('--rw_scale', type=int, default=1)
parser.add_argument('--replaybuffer_size', type=int, default=1000000)

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



def main():
    """ The main() function. """

    print("STARTING SPOT SAC")

    # TRAINING PARAMETERS
    seed = ARGS.Seed
    print("SEED: {}".format(seed))    
    
    max_timesteps = 4e7
    batch_size = 256
    eval_freq = 10
    save_model = True
    file_name = "spot_sac_"
    if len(ARGS.agent_name) !=0:
        file_name += ARGS.agent_name + "_"
    print("agentname:",file_name)
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


    print("first models path", models_path)
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    sacpath = models_path +"/SAC/" + file_name + rand_name[:-1] + "_s"+ str(seed)
    if not os.path.exists(sacpath):
        os.makedirs(sacpath)
    models_path = sacpath
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    print("models path", models_path)
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
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    print("STATE DIM: {}".format(state_dim))
    action_dim = env.action_space.shape[0]
    print("ACTION DIM: {}".format(action_dim))
    max_action = float(env.action_space.high[0])

    print("RECORDED MAX ACTION: {}".format(max_action))

    hidden_dim = ARGS.hidden_dim
    policy = PolicyNetwork(state_dim, action_dim, hidden_dim, episode_steps = ARGS.max_steps)

    replay_buffer_size = ARGS.replaybuffer_size
    replay_buffer = ReplayBuffer(replay_buffer_size)

    sac = SoftActorCritic(policy=policy,
                          state_dim=state_dim,
                          action_dim=action_dim,
                          replay_buffer=replay_buffer,
                          hidden_dim = hidden_dim,
                          policy_lr=ARGS.learningrate,
                          soft_q_lr=ARGS.learningrate,
                          ent_coef_lr=ARGS.learningrate)


    agent_path = models_path + "/" + str(file_name) + rand_name + "seed" + str(seed) + "_"+ str(ARGS.AgentNum)
    print("agent path:",agent_path)
    if os.path.exists(agent_path + "_policy_net"):
        print("Loading Existing agent")
        sac.load(agent_path)
        sac.policy_net.episode_steps = ARGS.max_steps
        policy = sac.policy_net

    # Evaluate untrained policy and init list for storage
    evaluations = []

    state = env.reset()
    done = False
    episode_reward = 0
    episode_reward_compare = []


    episode_timesteps = 0
    episode_num = 0


    env_step_size = env._time_step
    # State Machine for Random Controller Commands
    bz_step = BezierStepper(dt=env_step_size)

    # Bezier Gait Generator
    bzg = BezierGait(dt=env_step_size)

    # Initialize Normalizer
    normalizer = Normalizer(state_dim)

    # Spot Model
    spot = SpotModel()
    T_bf = copy.deepcopy(spot.WorldToFoot)
    T_bf0 = copy.deepcopy(spot.WorldToFoot)
    if ARGS.use_alpha20s:
        action = env.action_space.sample()
        action[:] = 0.0
        old_act = action[:actions_to_filter]



    BaseClearanceHeight = bz_step.ClearanceHeight
    BasePenetrationDepth = bz_step.PenetrationDepth

    print("STARTED SPOT SAC")

    for t in range(int(max_timesteps)):

        pos, orn, StepLength, LateralFraction, YawRate, StepVelocity, ClearanceHeight, PenetrationDepth = bz_step.StateMachine()
        env.spot.GetExternalObservations(bzg, bz_step)

        # Read UPDATED state based on controls and phase
        state = env.return_state()
        normalizer.observe(state)
        # NOTE: Don't normalize contacts - must stay 0/1
        state = normalizer.normalize(state)
        action = sac.policy_net.get_action(state)
        action_output = copy.deepcopy(action)
        contacts = state[-4:]
        
        if ARGS.use_alpha20s:
            # EXP FILTER
            action[:actions_to_filter] = alpha * old_act + (
                1.0 - alpha) * action[:actions_to_filter]
            old_act = action[:actions_to_filter]

        ClearanceHeight += action[0] * CD_SCALE

        # CLIP EVERYTHING
        StepLength = np.clip(StepLength, bz_step.StepLength_LIMITS[0],
                             bz_step.StepLength_LIMITS[1])
        StepVelocity = np.clip(StepVelocity, bz_step.StepVelocity_LIMITS[0],
                               bz_step.StepVelocity_LIMITS[1])
        LateralFraction = np.clip(LateralFraction,
                                  bz_step.LateralFraction_LIMITS[0],
                                  bz_step.LateralFraction_LIMITS[1])
        YawRate = np.clip(YawRate, bz_step.YawRate_LIMITS[0],
                          bz_step.YawRate_LIMITS[1])
        ClearanceHeight = np.clip(ClearanceHeight,
                                  bz_step.ClearanceHeight_LIMITS[0],
                                  bz_step.ClearanceHeight_LIMITS[1])
        PenetrationDepth = np.clip(PenetrationDepth,
                                   bz_step.PenetrationDepth_LIMITS[0],
                                   bz_step.PenetrationDepth_LIMITS[1])

        # For auto yaw control
        yaw = env.return_yaw()
        YawRate += -yaw * P_yaw

        if ARGS.use_alpha20s:
            # Get Desired Foot Poses
            if episode_timesteps > 20:
                T_bf = bzg.GenerateTrajectory(StepLength, LateralFraction,
                                                YawRate, StepVelocity, T_bf0,
                                                T_bf, ClearanceHeight,
                                                PenetrationDepth, contacts)
            else:
                T_bf = bzg.GenerateTrajectory(0.0, 0.0, 0.0, 0.1, T_bf0,
                                                T_bf, ClearanceHeight,
                                                PenetrationDepth, contacts)
                action[:] = 0.0
        else:
            # Get Desired Foot Poses
            T_bf = bzg.GenerateTrajectory(StepLength, LateralFraction, YawRate,
                                        StepVelocity, T_bf0, T_bf,
                                        ClearanceHeight, PenetrationDepth,
                                      contacts)
        action[2:] *= RESIDUALS_SCALE

        # same as ARS:
        T_bf_copy = copy.deepcopy(T_bf)
        T_bf_copy["FL"][:3, 3] += action[2:5]
        T_bf_copy["FR"][:3, 3] += action[5:8]
        T_bf_copy["BL"][:3, 3] += action[8:11]
        T_bf_copy["BR"][:3, 3] += action[11:14]


        pos[2] += abs(action[1]) * Z_SCALE
        joint_angles = spot.IK(orn, pos, T_bf)
        # Pass Joint Angles
        env.pass_joint_angles(joint_angles.reshape(-1))

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done)
        episode_timesteps += 1

        # Store data in replay buffer
        replay_buffer.push(state, action_output, reward*ARGS.rw_scale, next_state, done_bool)

        state = next_state
        episode_reward += reward
        episode_reward_compare.append(reward)

        # Train agent after collecting sufficient data for buffer
        if len(replay_buffer) > batch_size:
            sac.soft_q_update(batch_size)

        if episode_timesteps > sac.policy_net.episode_steps:
            done = True

        if done:
            # Reshuffle State Machine
            bzg.reset()
            bz_step.reshuffle()
            bz_step.ClearanceHeight = BaseClearanceHeight
            bz_step.PenetrationDepth = BasePenetrationDepth
            # +1 to account for 0 indexing.
            # +0 on ep_timesteps since it will increment +1 even if done=True
            compare_reward = 0
            if len(episode_reward_compare) > 22:
                compare_cutoff = ARGS.max_steps - 20 # don't use the last 20 steps for compare
                compare_reward = np.mean(np.array(episode_reward_compare[:compare_cutoff])) 
            # Store Results (concat)
            if episode_num == 0:
                res = np.array(
                    [[episode_reward, episode_reward / float(episode_timesteps), compare_reward]])
            else:
                new_res = np.array(
                    [[episode_reward, episode_reward / float(episode_timesteps), compare_reward]])
                res = np.concatenate((res, new_res))
            np.save(
                results_path + "/" + str(file_name) + rand_name + "seed" +
                str(seed), res)
            print(
                "Total T: {} Episode Num: {} Episode T: {} Reward: {:.2f} REWARD PER STEP: {:.2f} Rew Comp: {:.2f}"
                .format(t + 1, episode_num, episode_timesteps, episode_reward,
                        episode_reward / float(episode_timesteps), compare_reward))
            # Reset environment
            state, done = env.reset(), False
            evaluations.append(episode_reward)
            episode_reward = 0
            episode_reward_compare = []

            episode_timesteps = 0
            episode_num += 1
            
            # Evaluate episode
            if (episode_num + 1) % eval_freq == 0 or episode_num > ARGS.num_epochs:
                if save_model:
                    sac.save(models_path + "/" + str(file_name) + rand_name + "seed" +
                    str(seed) + "_"+ str(episode_num))

            if episode_num > ARGS.num_epochs :
                break

    env.close()

if __name__ == '__main__':
    main()