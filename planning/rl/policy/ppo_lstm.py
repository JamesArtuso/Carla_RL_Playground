import gym
import numpy as np
import torch
import torch.nn as nn
from planning.rl.policy.base_policy import Policy
from planning.rl.policy.pid_controller import LagrangianPIDController
from planning.rl.policy.model.lstm_ac import (LSTMGaussianActor, LSTMCritic)
from planning.rl.util.logger import EpochLogger
from planning.rl.util.torch_util import (count_vars, get_device_name, to_device,
                                     to_ndarray, to_tensor)
from torch.optim import Adam
from planning.rl.policy.image_encoder_wrapper import ImageEncoderWrapperLSTM, CNN
import cv2

class PPO(Policy):
    def __init__(self,
                 env: gym.Env,
                 logger: EpochLogger,
                 actor_lr=0.0003,
                 critic_lr=0.001,
                 obs_type=0,
                 hidden_sizes=[256, 256],
                 clip_ratio=0.2,
                 target_kl=0.01,
                 train_actor_iters=80,
                 train_critic_iters=80,
                 gamma=0.97,
                 memory_size = 32,
                 burnin = 16,
                 obs_size=4,
                 **kwargs) -> None:
        r'''
        Promximal Policy Optimization (PPO)

        Args:
        @param env : The environment must satisfy the OpenAI Gym API.
        @param logger: Log useful informations, and help to save model
        @param actor_lr, critic_lr (float): Learning rate for policy and Q-value learning.
        @param obs_type: the actor critic model name

        @param clip_ratio (float): Hyperparameter for clipping in the policy objective. Roughly: how far can the new policy go from the old policy while still profiting (improving the objective function)? The new policy can still go farther than the clip_ratio says, but it doesn't help on the objective anymore. (Usually small, 0.1 to 0.3.) Typically denoted by :math:`\epsilon`. 
        @param target_kl (float): Roughly what KL divergence we think is appropriate between new and old policies after an update. This will get used for early stopping. (Usually small, 0.01 or 0.05.)
        @param train_actor_iters, train_critic_iters (int): Training iterations for actor and critic
        '''
        super().__init__()
        self.logger = logger
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.train_actor_iters = train_actor_iters
        self.train_critic_iters = train_critic_iters
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.hidden_sizes = hidden_sizes

        ################ create actor critic model ###############
        #self.obs_dim = env.observation_space.shape[0]
        self.obs_dim = obs_size
        self.act_dim = 2
        #self.act_dim = env.action_space.shape[0]
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        #self.act_lim = env.action_space.high[0]
        self.act_lim = 1
        self.obs_type = obs_type

        if obs_type == 0:
            if isinstance(env.action_space, gym.spaces.Box):
                actor = LSTMGaussianActor(self.obs_dim, self.act_dim,
                                         -self.act_lim, self.act_lim,
                                         hidden_sizes, nn.ReLU)
            critic = LSTMCritic(self.obs_dim,hidden_sizes, nn.ReLU)
        elif obs_type== 1:
            if isinstance(env.action_space, gym.spaces.Box):
                actor_backend = LSTMGaussianActor(self.obs_dim, self.act_dim,
                                         -self.act_lim, self.act_lim,
                                         hidden_sizes, nn.ReLU)
            actor_encoder = CNN(self.obs_dim)
            critic_backend = LSTMCritic(self.obs_dim, hidden_sizes, nn.ReLU)
            critic_encoder = CNN(self.obs_dim)
            actor = ImageEncoderWrapperLSTM(actor_encoder, actor_backend)
            critic = ImageEncoderWrapperLSTM(critic_encoder, critic_backend)
        else:
            raise ValueError(f"{obs_type} obs_type does not support.")

        #actor = LSTMGaussianActor(self.obs_dim, self.act_dim,
        #                            -self.act_lim, self.act_lim,
        #                            hidden_sizes, nn.ReLU)
        
        #critic = LSTMCritic(self.obs_dim,hidden_sizes, nn.ReLU)

        # Set up optimizer and device
        self._ac_training_setup(actor, critic)

        # Set up model saving
        #self.save_model()

        # Count variables
        var_counts = tuple(
            count_vars(module) for module in [self.actor, self.critic])
        logger.log('\nNumber of parameters: \t actor: %d, \t critic: %d\n' %
                   var_counts)

    def _ac_training_setup(self, actor, critic):
        self.actor, self.critic = to_device([actor, critic], get_device_name())

        # Set up optimizers for policy and value function
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(),
                                     lr=self.critic_lr)

    def act(self, obs, deterministic=False):
        '''
        Given a single obs, return the action, value, logp.
        This API is used to interact with the env.

        @param obs, 1d ndarray
        @param eval, evaluation mode
        @return act, value, logp, 1d ndarray
        '''
        obs = to_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            _, a, logp_a = self.actor_forward(obs, deterministic=deterministic)
            v = self.critic_forward(self.critic, obs)
        # squeeze them to the right shape
        a, v, logp_a = np.squeeze(to_ndarray(a), axis=0), np.squeeze(
            to_ndarray(v)), np.squeeze(to_ndarray(logp_a))
        return a, v, logp_a

    def learn_on_batch(self, data: dict):
        '''
        Given a batch of data, train the policy
        data keys: (obs, act, ret, adv, logp)
        '''
        self._update_actor(data)

        LossV, DeltaLossV = self._update_critic(self.critic, data["obs"],
                                                data["ret"],
                                                data['mask'])
        # Log critic update info
        self.logger.store(LossV=LossV, DeltaLossV=DeltaLossV)

    def critic_forward(self, critic, obs):
        # Critical to ensure value has the right shape.
        # Without this, the training stability will be greatly affected!
        # For instance, shape [3] - shape[3,1] = shape [3, 3] instead of shape [3]
        return torch.squeeze(critic(obs), -1)

    def actor_forward(self, obs, act=None, deterministic=False):
        r''' 
        Return action distribution and action log prob [optional].
        @param obs, [tensor], (batch, obs_dim)
        @param act, [tensor], (batch, act_dim). If None, log prob is None
        @return pi, [torch distribution], (batch,)
        @return a, [torch distribution], (batch, act_dim)
        @return logp, [tensor], (batch,)
        '''
        pi, a, logp = self.actor(obs, act, deterministic)
        return pi, a, logp
    


    def _update_actor(self, data):
        '''
        Update the actor network
        '''
        obs, act, adv, logp_old, mask = to_tensor(data['obs']), to_tensor(
            data['act']), to_tensor(data['adv']), to_tensor(data['logp']), to_tensor(data['mask'])

        def policy_loss():

            pi, mu, logp, _ = self.actor.forward_sequential(obs, act)

            ratio = torch.exp(logp - logp_old)

            clipped_ratio = torch.clamp(
                ratio,
                1 - self.clip_ratio,
                1 + self.clip_ratio
            )

            loss_unreduced = -torch.min(
                ratio * adv,
                clipped_ratio * adv
            )

            # Mask burn-in + padding
            loss_pi = (loss_unreduced * mask).sum() / mask.sum()

            # Masked KL
            approx_kl = ((logp_old - logp) * mask).sum() / mask.sum()

            # Masked entropy
            ent = (pi.entropy() * mask.unsqueeze(-1)).sum() / mask.sum()

            clipped = (ratio.gt(1 + self.clip_ratio) |
                    ratio.lt(1 - self.clip_ratio)).float()

            clipfrac = (clipped * mask).sum() / mask.sum()

            pi_info = dict(
                KL=approx_kl.item(),
                Entropy=ent.item(),
                ClipFrac=clipfrac.item()
            )

            return loss_pi, pi_info

        pi_l_old, pi_info_old = policy_loss()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_actor_iters):
            self.actor_optimizer.zero_grad()
            loss_pi, pi_info = policy_loss()
            if i == 0 and pi_info['kl'] >= 1e-7:
                print("**" * 20)
                print("1st kl: ", pi_info['kl'])
            if pi_info['kl'] > 1.5 * self.target_kl:
                self.logger.log(
                    'Early stopping at step %d due to reaching max kl.' % i)
                break
            loss_pi.backward()
            self.actor_optimizer.step()

        # Log actor update info
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']

        self.logger.store(StopIter=i,
                          LossPi=to_ndarray(pi_l_old),
                          KL=kl,
                          Entropy=ent,
                          ClipFrac=cf,
                          DeltaLossPi=(to_ndarray(loss_pi) -
                                       to_ndarray(pi_l_old)))

    def _update_critic(self, critic, obs, ret, mask):

        obs  = to_tensor(obs)
        ret  = to_tensor(ret)
        mask = to_tensor(mask)

        def critic_loss():

            values, _ = critic.forward_sequential(obs)
            values = torch.squeeze(values, -1)
            #print(f'Values: {values.shape}')
            #print(f'ret: {ret.shape}')
            # MSE per timestep
            loss_unreduced = (values - ret) ** 2

            # Mask burn-in and padding
            loss = (loss_unreduced * mask).sum() / mask.sum()

            return loss

        loss_old = critic_loss().item()

        for _ in range(self.train_critic_iters):
            self.critic_optimizer.zero_grad()
            loss = critic_loss()
            loss.backward()
            self.critic_optimizer.step()

        return loss_old, loss.item() - loss_old

    def save_model(self):
        self.logger.setup_pytorch_saver((self.actor, self.critic))

    def load_model(self, path):
        actor, critic = torch.load(path)
        self._ac_training_setup(actor, critic)
        # Set up model saving
        print('loaded!')
        print(actor)
        print('\n\n\n')
        print(critic)
        self.save_model()

    def reset_hidden(self):
        self.actor.reset_hidden(1)
        self.critic.reset_hidden(1)


class PPOLagrangian(PPO):
    def __init__(self,
                 env: gym.Env,
                 logger: EpochLogger,
                 cost_limit=40,
                 timeout_steps=200,
                 KP=0,
                 KI=0.1,
                 KD=0,
                 per_state=False,
                 **kwargs) -> None:
        r'''
        Proximal Policy Optimization (PPO) with Lagrangian multiplier
        '''
        super().__init__(env, logger, **kwargs)

        self.cost_limit = cost_limit
        self.qc_thres = cost_limit * (1 - self.gamma**timeout_steps) / (
            1 - self.gamma) / timeout_steps
        print("Cost constraint: ", self.qc_thres)

        self.controller = LagrangianPIDController(KP, KI, KD, self.cost_limit,
                                                  per_state)

        #self.qc_critic = to_device(
        #    mlp([self.obs_dim] + list(self.hidden_sizes) + [1], nn.ReLU))
        if(self.obs_type == 0):
            self.qc_critic = to_device(LSTMCritic(self.obs_dim, 256, nn.ReLU))
        elif(self.obs_type ==1):
            qc_encoder = CNN(self.obs_dim)
            qc_backend= LSTMCritic(self.obs_dim, 256, nn.ReLU)
            self.qc_critic = to_device(ImageEncoderWrapperLSTM(qc_encoder, qc_backend))

        self.qc_critic_optimizer = Adam(self.qc_critic.parameters(),
                                        lr=self.critic_lr)
        self.save_model()

    def learn_on_batch(self, data: dict):
        super().learn_on_batch(data)
        LossVQC, DeltaLossVQC = self._update_critic(self.qc_critic,
                                                    data["obs"],
                                                    data["cost_ret"],
                                                    data['mask'])
        # Log safety critic update info
        self.logger.store(LossVQC=LossVQC, DeltaLossVQC=DeltaLossVQC)

    def post_epoch_process(self):
        '''
        do nothing.
        '''
        pass

    def get_qc_v(self, obs):
        obs = to_tensor(obs).unsqueeze(0)#.reshape(1, -1)
        with torch.no_grad():
            v = self.critic_forward(self.qc_critic, obs)
        return np.squeeze(to_ndarray(v))

    def _update_actor(self, data):

        obs       = to_tensor(data['obs'])
        act       = to_tensor(data['act'])
        adv       = to_tensor(data['adv'])
        cost_adv  = to_tensor(data['cost_adv'])
        logp_old  = to_tensor(data['logp'])
        mask      = to_tensor(data['mask'])
        ep_cost   = to_tensor(data['ep_cost'])

        # Multiplier must NOT receive gradients
        multiplier = self.controller.control(ep_cost).detach()

        def policy_loss():

            # Sequential forward
            pi, mu, logp, _ = self.actor.forward_sequential(obs, act)

            ratio = torch.exp(logp - logp_old)

            clipped_ratio = torch.clamp(
                ratio,
                1 - self.clip_ratio,
                1 + self.clip_ratio
            )

            # PPO objective
            ppo_obj = torch.min(
                ratio * adv,
                clipped_ratio * adv
            )

            # Cost penalty (NO clipping — matches standard Lagrangian PPO)
            cost_penalty = ratio * cost_adv * multiplier

            # Combine
            loss_unreduced = -(ppo_obj - cost_penalty)

            # Mask burn-in + padding
            loss_pi = (loss_unreduced * mask).sum() / mask.sum()

            # Normalize by multiplier scale (as in your original)
            loss_pi = loss_pi / (1 + multiplier)

            # ----- Logging metrics (masked) -----

            approx_kl = ((logp_old - logp) * mask).sum() / mask.sum()

            entropy = pi.entropy().sum(-1)
            ent = (entropy * mask).sum() / mask.sum()

            clipped = (
                ratio.gt(1 + self.clip_ratio) |
                ratio.lt(1 - self.clip_ratio)
            ).float()

            clipfrac = (clipped * mask).sum() / mask.sum()

            pi_info = dict(
                KL=approx_kl.item(),
                Entropy=ent.item(),
                ClipFrac=clipfrac.item(),
                Multiplier=multiplier.item()
            )

            return loss_pi, pi_info

        pi_l_old, pi_info_old = policy_loss()

        for i in range(self.train_actor_iters):

            self.actor_optimizer.zero_grad()
            loss_pi, pi_info = policy_loss()

            if pi_info['KL'] > 1.5 * self.target_kl:
                self.logger.log(
                    f'Early stopping at step {i} due to max KL.'
                )
                break

            loss_pi.backward()
            self.actor_optimizer.step()

        self.logger.store(
            StopIter=i,
            LossPi=pi_l_old.item(),
            DeltaLossPi=loss_pi.item() - pi_l_old.item(),
            ObservedCost=ep_cost.mean().item(),
            Lagrangian=multiplier.item(),
            **pi_info
        )
    def reset_hidden(self):
        self.actor.reset_hidden(1)
        self.critic.reset_hidden(1)
        self.qc_critic.reset_hidden(1)
    def save_model(self):
        self.logger.setup_pytorch_saver((self.actor, self.critic, self.qc_critic))        


    def load_model(self, path):
        try:
            actor, critic, qc_critic = torch.load(path)
            self._acq_training_setup(actor, critic, qc_critic)
        except:
            actor, critic = torch.load(path)
            self._ac_training_setup(actor, critic)
        # Set up model saving
        print('loaded!')
        print(actor)
        print('\n\n\n')
        print(critic)
        self.save_model()

    def _acq_training_setup(self, actor, critic, qc_critic):
        self.actor, self.critic, self.qc_critic = to_device([actor, critic, qc_critic], get_device_name())

        # Set up optimizers for policy and value function
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(),
                                     lr=self.critic_lr)
        self.qc_critic_optimizer = Adam(self.qc_critic.parameters(),
                                        lr=self.critic_lr)