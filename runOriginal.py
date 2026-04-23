import gym
import gym_carla
import carla
#from models.rl.ppo import PPO
from planning.safe_rl.policy.ppo import PPOLagrangian
from models.rl.sac import SAC
import torch 
from planning.safe_rl.util.logger import EpochLogger
import traceback
import os.path as osp

from planning.rl.util.run_util import load_config
from util.torch_util import set_seed, set_torch_variable
from util.replay_buffer import ReplayBuffer

if __name__ == '__main__':
    
    #Agent Parameters
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--output_dir', type=str, default='log')
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
    args = parser.parse_args()
    args_dict = vars(args)

    err_list = []
    for agent_cfg in args.agent_cfg:
        for env_cfg in args.env_cfg:
            set_torch_variable(args.device)
            torch.set_num_threads(args.threads)
            set_seed(args.seed)
            
            agent_config_path = osp.join(args.ROOT_DIR, 'carla_gym/models/config', agent_cfg)
            env_config_path = osp.join(args.ROOT_DIR, 'carla_gym/env_configs', env_cfg)
            agent_config = load_config(agent_config_path)
            env_config = load_config(env_config_path)
            
            agent_config.update(args_dict)
        
    agent_config['mode'] = 'train_agent'
    agent_config['ego_action_dim'] = 2
    agent_config['ego_state_dim'] = 4
    agent_config['ego_action_limit'] = 1.0
    
    
    env = gym.make('carla-v0', params=env_config)
    obs = env.reset()
    print(env.observation_space)
    logger = EpochLogger()
    test_model = PPOLagrangian(env, logger) 
    test_model.load_model('/home/jaart/Downloads/model.pt')
    num_episodes = 1000
    steps_per_episode = 500
    buff = ReplayBuffer(capacity=2048)
    frame_skip = 1
    for ep in range(num_episodes):
        obs = env.reset()
        state = obs['state']
        episode_reward = 0
        for t in range(int(steps_per_episode/frame_skip)):
            action, v, logp = test_model.act(
                obs=state,
                deterministic=True
            )
            
            #print(f' Throttle: {action[0]} | Steer: {action[1]}')
            reward = 0
            for i in range(frame_skip):
                obs, r, done, info = env.step(action)
                reward += r
                if(done):
                    break
            
            next_state = obs['state']

            buff.store(
                state = state,
                action = action,
                reward = reward/frame_skip,
                next_state=next_state,
                done=done
            )
            state = next_state
            episode_reward += reward
            
            if done:
                break
        print(f'Episode {ep} | total reward: {episode_reward:.2f} | steps: {t+1}')
        
        if len(buff) >= test_model.rollout_steps:
            test_model.train(buff)
            #test_model.save_model(16) #12 is pretty good
            buff.reset_buffer()

