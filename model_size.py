import gym
import gym_carla
import carla
from planning.rl.policy.ppo import PPOLagrangian as PPO_MLP
from planning.rl.policy.ppo_lstm import PPOLagrangian as PPO_lstm
from planning.rl.util.run_util import load_config
import os.path as osp
from planning.rl.util.torch_util import set_torch_variable
import torch 
from planning.rl.util.logger import EpochLogger

def print_model_info(name, model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name}:")
    print(f"  Total params: {total_params}")
    print(f"  Trainable params: {trainable_params}")
    #print("  Layers:")
    #for i, (n, p) in enumerate(model.named_parameters()):
    #    print(f"    {i+1}. {n}: {tuple(p.shape)}")
    print()


logger = EpochLogger('carla_gym/notit', 'progress.txt', 'ppo',eval_mode= False, use_tensor_board = True, resume = False)



root_dir = osp.abspath(osp.dirname(osp.dirname(osp.realpath(__file__))))
env_config_path = osp.join(root_dir, 'carla_gym/env_configs', 'dummy.yaml')
env_config = load_config(env_config_path)
set_torch_variable('cuda:0')
torch.set_num_threads(4)

mlp = osp.join(root_dir, 'carla_gym/planning/config', 'ppo.yaml')
lstm = osp.join(root_dir, 'carla_gym/planning/config', 'ppo_lstm.yaml')
mlp_bev = osp.join(root_dir, 'carla_gym/planning/config', 'ppo_bev.yaml')
lstm_bev = osp.join(root_dir, 'carla_gym/planning/config', 'ppo_lstm_bev.yaml')

mlp = load_config(mlp)
lstm = load_config(lstm)
mlp_bev = load_config(mlp_bev)
lstm_bev = load_config(lstm_bev)

env = gym.make('carla-v0', params=env_config)

mlp = PPO_MLP(env, logger, **mlp['ppo'])
mlp.load_model(osp.join(root_dir, 'carla_gym/carla_gym/PPO_1000/model_save', 'model.pt'))
lstm = PPO_lstm(env, logger, **lstm['ppo'])
lstm.load_model(osp.join(root_dir, 'carla_gym/carla_gym/PPO_LSTM_1000/model_save', 'model.pt'))
mlp_bev = PPO_MLP(env, logger, **mlp_bev['ppo'])
mlp_bev.load_model(osp.join(root_dir, 'carla_gym/carla_gym/PPO_BEV_1000/model_save', 'model.pt'))
lstm_bev = PPO_lstm(env, logger, **lstm_bev['ppo'])
lstm_bev.load_model(osp.join(root_dir, 'carla_gym/carla_gym/LSTM_PPO_BEV_RUN_2_FINAl/model_save', 'model.pt'))


# --- MLP PPO ---
print_model_info("MLP Actor", mlp.actor)
print_model_info("MLP Critic", mlp.critic)
print_model_info("MLP QC Critic", mlp.qc_critic)

# --- LSTM PPO ---
print_model_info("LSTM Actor", lstm.actor)
print_model_info("LSTM Critic", lstm.critic)
print_model_info("LSTM QC Critic", lstm.qc_critic)

# --- MLP BEV PPO ---
print_model_info("MLP BEV Actor Encoder", mlp_bev.actor.encoder)
print_model_info("MLP BEV Actor Model", mlp_bev.actor.model)
print_model_info("MLP BEV Critic", mlp_bev.critic.model)
print_model_info("MLP BEV QC Critic", mlp_bev.qc_critic.model)

# --- LSTM BEV PPO ---
print_model_info("LSTM BEV Actor Encoder", lstm_bev.actor.encoder)
print_model_info("LSTM BEV Actor", lstm_bev.actor.model)
print_model_info("LSTM BEV Critic", lstm_bev.critic.model)
print_model_info("LSTM BEV QC Critic", lstm_bev.qc_critic.model)