# From https://github.com/wjh720/QPLEX/, added here for convenience.
import copy
import time

import torch as th
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from modules.mixers.dmaq_general import DMAQer
from utils.rl_utils import build_td_lambda_targets
from utils.th_utils import get_parameters_num


class DMAQ_qattenLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "dmaq":
                self.mixer = DMAQer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = Adam(params=self.params, lr=args.lr)

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.n_actions = self.args.n_actions
        self.train_t = 0
        self.avg_time = 0

    def sub_train(self, batch: EpisodeBatch, t_env: int, episode_num: int, mac, mixer, optimiser, params,
                  save_data=None, split_data=False):
        start_time = time.time()

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]

        # Calculate estimated Q-Values
        self.mac.set_train_mode()
        mac_out = []
        mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        with th.no_grad():
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            max_action_qvals, cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)  # [bs, traj, n_agent, 1]

            # %%%%%%%%%%%%%%%%%%%%%%%%%% Calculate the Q-Values necessary for the target %%%%%%%%%%%%%%%%%%%%%%%%%%
            # Set target mac to testing mode
            self.target_mac.set_evaluation_mode()
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)

            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

            # Max over target Q-Values
            assert self.args.double_q
            # Get actions that maximise live Q (for double q-learning)
            target_chosen_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Mixer %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Set mixing net to training mode
        mixer.train()
        # %%%%%%%%%%%%%%%%%%%%%%% Current \sum_i{Q_i} %%%%%%%%%%%%%%%%%%%%%%%%
        ans_chosen = mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True)
        # %%%%%%%%%%%%%%%%%%%%%%% Current \sum_i{adv_i} %%%%%%%%%%%%%%%%%%%%%%%%
        ans_adv = mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot,
                        max_q_i=max_action_qvals[:, :-1].contiguous().squeeze(3), is_v=False)

        chosen_action_qvals = ans_chosen + ans_adv

        with th.no_grad():
            self.target_mixer.eval()
            # %%%%%%%%%%%%%%%%%%%%%%% Target \sum_i{Q_i} %%%%%%%%%%%%%%%%%%%%%%%%
            target_chosen = self.target_mixer(target_chosen_qvals, batch["state"], is_v=True)

            # %%%%%%%%%%%%%%%%%%%%%%% Target \sum_i{adv_i} %%%%%%%%%%%%%%%%%%%%%%%%
            # Mask out unavailable actions
            target_mac_out[avail_actions == 0] = -9999999
            target_max_qvals = target_mac_out.max(dim=3)[0]
            # Onehot target actions
            cur_max_actions_onehot = th.zeros(cur_max_actions.squeeze(3).shape + (self.n_actions,),
                                              device=cur_max_actions.device)
            cur_max_actions_onehot = cur_max_actions_onehot.scatter_(3, cur_max_actions, 1)
            target_adv = self.target_mixer(target_chosen_qvals, batch["state"],
                                           actions=cur_max_actions_onehot, max_q_i=target_max_qvals, is_v=False)

            target_max_qvals = target_chosen + target_adv

        # Calculate 1-step Q-Learning targets
        targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals,
                                          self.args.gamma, self.args.td_lambda)

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = 0.5 * (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        if not split_data:
            optimiser.zero_grad()

        loss.backward()

        if not split_data:
            grad_norm = th.nn.utils.clip_grad_norm_(params, self.args.grad_norm_clip)
            optimiser.step()

        self.train_t += 1
        self.avg_time += (time.time() - start_time - self.avg_time) / self.train_t
        print("Avg cost {} seconds".format(self.avg_time))

        if not split_data and t_env - self.log_stats_t >= self.args.learner_log_interval:
            with th.no_grad():
                is_max_action = (cur_max_actions[:, :-1] == actions).int().float()
                masked_hit_prob = th.mean(is_max_action, dim=2) * mask
                hit_prob = masked_hit_prob.sum() / mask.sum()
                self.logger.log_stat("loss", loss.item(), t_env)
                self.logger.log_stat("hit_prob", hit_prob.item(), t_env)
                self.logger.log_stat("grad_norm", grad_norm, t_env)
                mask_elems = mask.sum().item()
                self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
                self.logger.log_stat("q_taken_mean",
                                     (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents),
                                     t_env)
                self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                     t_env)
            self.log_stats_t = t_env

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, save_data=None):
        if self.args.use_cuda and str(self.mac.get_device()) == "cpu":
            self.mac.cuda()

        if self.args.n_agents > 20:
            split_num = 2
            a, b, c, d = batch.split(split_num)

            # Optimise
            self.optimiser.zero_grad()

            self.sub_train(a, t_env, episode_num, self.mac, self.mixer, self.optimiser, self.params,
                           save_data=save_data, split_data=True)
            del a

            self.sub_train(b, t_env, episode_num, self.mac, self.mixer, self.optimiser, self.params,
                           save_data=save_data, split_data=True)
            del b

            self.sub_train(c, t_env, episode_num, self.mac, self.mixer, self.optimiser, self.params,
                           save_data=save_data, split_data=True)
            del c

            self.sub_train(d, t_env, episode_num, self.mac, self.mixer, self.optimiser, self.params,
                           save_data=save_data, split_data=True)
            del d

            # Optimise
            grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
            self.optimiser.step()

        else:
            self.sub_train(batch, t_env, episode_num, self.mac, self.mixer, self.optimiser, self.params,
                           save_data=save_data, split_data=False)

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.target_mixer.load_state_dict(th.load("{}/mixer.th".format(path),
                                                      map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
