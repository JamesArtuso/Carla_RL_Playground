import gym
import gym_carla
import carla
#from models.rl.ppo import PPO
from planning.safe_rl.policy.ppo_lstm import PPOLagrangian
from models.rl.sac import SAC
import torch 
from planning.safe_rl.util.logger import EpochLogger
from planning.safe_rl.worker.on_policy_worker import OnPolicySequentialWorker

import traceback
import os.path as osp

from planning.rl.util.run_util import load_config
from util.torch_util import set_seed, set_torch_variable
from util.replay_buffer import ReplayBuffer
import time

def _log_metrics(epoch, total_steps, logger, time=None, verbose=True, cost_limit = 1e3):
    logger.log_tabular('CostLimit', cost_limit)
    logger.log_tabular('Epoch', epoch)
    logger.log_tabular('TotalEnvInteracts', total_steps)
    for key in logger.logger_keys:
        logger.log_tabular(key, average_only=True)
    if time is not None:
        logger.log_tabular('Time', time)
    # data_dict contains all the keys except Epoch and TotalEnvInteracts
    data_dict = logger.dump_tabular(
        x_axis="TotalEnvInteracts",
        verbose=verbose,
    )
    return data_dict







if __name__ == '__main__':
    
    #Agent Parameters
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--output_dir', type=str, default='carla_gym/experiments')
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.realpath(__file__)))))

    parser.add_argument('--max_episode_step', type=int, default=300)
    parser.add_argument('--auto_ego', action='store_true')
    parser.add_argument('--mode', '-m', type=str, default='eval', choices=['train_agent', 'train_scenario', 'eval'])
    parser.add_argument('--agent_cfg', nargs='*', type=str, default='dummy.yaml')
    parser.add_argument('--env_cfg', nargs='*', type=str, default='dummy.yaml')
    parser.add_argument('--continue_agent_training', '-cat', type=bool, default=False)
    parser.add_argument('--continue_scenario_training', '-cst', type=bool, default=False)

    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')   

    parser.add_argument('--num_scenario', '-ns', type=int, default=2, help='num of scenarios we run in one episode')
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--render', type=bool, default=True)
    parser.add_argument('--frame_skip', '-fs', type=int, default=1, help='skip of frame in each step')
    parser.add_argument('--port', type=int, default=2000, help='port to communicate with carla')
    parser.add_argument('--tm_port', type=int, default=8000, help='traffic manager port')
    parser.add_argument('--fixed_delta_seconds', type=float, default=0.1)
    args = parser.parse_args()
    args_dict = vars(args)

    err_list = []
    for agent_cfg in args.agent_cfg:
        for env_cfg in args.env_cfg:
            set_torch_variable(args.device)
            torch.set_num_threads(args.threads)
            set_seed(args.seed)
            
            agent_config_path = osp.join(args.ROOT_DIR, 'carla_gym/planning/config', agent_cfg)
            env_config_path = osp.join(args.ROOT_DIR, 'carla_gym/env_configs', env_cfg)
            agent_config = load_config(agent_config_path)
            env_config = load_config(env_config_path)
            
            agent_config.update(args_dict)
        
    agent_config['mode'] = 'train_agent'
    agent_config['ego_action_dim'] = 2
    agent_config['ego_state_dim'] = 4
    agent_config['ego_action_limit'] = 1.0



    #Testing Setup
    num_epochs = 1000
    steps_per_episode = 500
    buff = ReplayBuffer(capacity=2048)
    frame_skip = 1
    save_freq = 50
    total_env_interactions = 0 
    start_epoch = 0
    #Env Setup
    env = gym.make('carla-v0', params=env_config)
    obs = env.reset()
    env.seed(5)

    #Resume Setup
    progress_path = osp.join(args.output_dir, 'progress.txt')
    resuming = osp.exists(progress_path)
    #resuming = True
    #Logger setup
    out_dir = osp.join(args.ROOT_DIR, 'carla_gym/experiments', agent_cfg)

    logger = EpochLogger(args.output_dir, 'progress.txt', 'ppo',eval_mode= False, use_tensor_board = True, resume = resuming)

    #Model Setup
    test_model = PPOLagrangian(env, logger, **agent_config['ppo']) 

    #Resuming
    if(resuming):
        data_dict = logger.load_progress()
        start_epoch = int(float(data_dict['Epoch'][-1])) + 1
        start_time = time.time() - int(float(data_dict['Time'][-1]))
        total_env_interactions = int(float(data_dict['TotalEnvInteracts'][-1]))
        logger.set_steps(total_env_interactions)
        logger.set_epoch(start_epoch)
        test_model.load_model(osp.join(args.ROOT_DIR, 'carla_gym/carla_gym/experiments/model_save', 'model.pt'))
        print(f'total env interactions: {total_env_interactions}')
    else:
        start_time = time.time()


    worker =  OnPolicySequentialWorker(env, test_model, logger , **agent_config['ppo']['worker_config'])
    

    for ep in range(start_epoch, num_epochs):
        epoch_steps = 0
        steps = worker.work()
        epoch_steps += steps
        total_env_interactions += epoch_steps
        data = worker.get_sample()
        #for k in data.keys():
        #    print(f'{k}: {data[k].shape}')
        test_model.learn_on_batch(data)
        for _ in range(1):
            worker.eval()
        if hasattr(test_model, "post_epoch_process"):
            test_model.post_epoch_process()
        if(ep % save_freq == 0 or ep == num_epochs - 1):
            logger.save_state({'env': None}, None)
        data_dict = _log_metrics(ep, total_env_interactions, logger, time.time() - start_time,True, agent_config['ppo']['cost_limit'])