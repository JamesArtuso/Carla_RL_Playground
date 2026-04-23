import argparse
import gym
import gym_carla
import carla
#from models.rl.ppo import PPO
from planning.rl.policy.ppo import PPOLagrangian
from planning.rl.policy.ppo_lstm import PPOLagrangian as PPOLagrangianLSTM
import torch 
from planning.rl.util.logger import EpochLogger
from planning.rl.worker.on_policy_worker import OnPolicyWorker
from planning.rl.worker.on_policy_worker import OnPolicySequentialWorker
import os.path as osp

from planning.rl.util.run_util import load_config
from planning.rl.util.torch_util import seed_torch, set_torch_variable
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--output_dir', type=str, default='logs/experiments')
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
    parser.add_argument('--no_gui', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=1000)

    args = parser.parse_args()
    return args

def load_experiment_configs(args, agent_cfg, env_cfg):
    args_dict = vars(args)
    set_torch_variable(args.device)
    torch.set_num_threads(args.threads)
    seed_torch(args.seed)
    
    agent_config_path = osp.join(args.ROOT_DIR, 'carla_gym/planning/config', agent_cfg)
    env_config_path = osp.join(args.ROOT_DIR, 'carla_gym/env_configs', env_cfg)
    agent_config = load_config(agent_config_path)
    env_config = load_config(env_config_path)
    env_config['no_gui'] = args.no_gui
    
    agent_config.update(args_dict)
        
    agent_config['mode'] = 'train_agent'
    agent_config['ego_action_dim'] = 2
    agent_config['ego_state_dim'] = 4
    agent_config['ego_action_limit'] = 1.0
    return agent_config, env_config


def make_logger(args, agent_config):
    progress_path = osp.join(args.output_dir, 'progress.txt')
    resuming = osp.exists(progress_path)
    logger = EpochLogger(args.output_dir, 'progress.txt', agent_config['policy'], eval_mode= False, use_tensor_board = True, resume = resuming)
    
    return logger, resuming

def make_env(args, env_config):
    env = gym.make('carla-v0', params=env_config)
    env.seed(args.seed)
    return env

def make_agent(args, logger, agent_config, env):
    if(agent_config['policy'] == 'ppo'):
        if(agent_config['ac_model'] == 'mlp'):
            agent = PPOLagrangian(env, logger, **agent_config['ppo']) 
            worker =  OnPolicyWorker(env, agent, logger , **agent_config['ppo']['worker_config'], obs_type=agent_config['ppo']['obs_type'])
        elif(agent_config['ac_model'] == 'lstm'):
            agent = PPOLagrangianLSTM(env, logger, **agent_config['ppo']) 
            worker =  OnPolicySequentialWorker(env, agent, logger , **agent_config['ppo']['worker_config'], obs_type=agent_config['ppo']['obs_type'])
    return agent, worker


def train(args, agent_cfg, env_cfg):
    agent_config, env_config = load_experiment_configs(args, agent_cfg, env_cfg)

    logger, resuming = make_logger(args, agent_config)

    env = make_env(args, env_config)

    agent, worker = make_agent(args, logger, agent_config, env)
    #Resuming
    if(resuming):
        data_dict = logger.load_progress()
        start_epoch = int(float(data_dict['Epoch'][-1])) + 1
        total_env_interactions = int(float(data_dict['TotalEnvInteracts'][-1]))
        logger.set_steps(total_env_interactions)
        logger.set_epoch(start_epoch)
        start_time = time.time() - int(float(data_dict['Time'][-1]))
        agent.load_model(osp.join(args.ROOT_DIR, 'carla_gym/logs/experiments/model_save', 'model.pt'))
        print(f'total env interactions: {total_env_interactions}')
    else:
        start_time = time.time()
        start_epoch = 0
        total_env_interactions = 0
    for ep in range(start_epoch, args.epochs):
        epoch_steps = 0
        steps = worker.work()
        epoch_steps += steps
        total_env_interactions += epoch_steps
        data = worker.get_sample()
        agent.learn_on_batch(data)
        for _ in range(1):
            worker.eval()
        if hasattr(agent, "post_epoch_process"):
            agent.post_epoch_process()
        if(ep % 10 == 0 or ep == args.epochs - 1):
            logger.save_state({'env': None}, None)
        data_dict = _log_metrics(ep, total_env_interactions, logger, time.time() - start_time,True, agent_config['ppo']['cost_limit'])




if __name__ == '__main__':
    args = parse_args()

    for agent_cfg in args.agent_cfg:
        for env_cfg in args.env_cfg:
            train(args, agent_cfg, env_cfg)