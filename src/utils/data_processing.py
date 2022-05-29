#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from components.episode_buffer import EpisodeBatch
import copy
import numpy as np
import torch as th


def clear_no_reward_sub_trajectory(batch):
    """
    :param batch:
    :return:
    """
    filled = batch.data.transition_data["filled"]  # [bs, traj_length, 1]
    rewards = batch.data.transition_data["reward"]  # [bs, traj_length, 1]
    bs, traj_length = filled.shape[0], filled.shape[1]
    fixed_row = []
    for t in range(traj_length - 1, 0, -1):
        remained_rows = [i for i in range(0, bs) if i not in fixed_row]
        for row_idx in remained_rows:
            if rewards[row_idx, t - 1, 0] == 0:  # no reward
                filled[row_idx, t, 0] = 0
                if t == 1:
                    filled[row_idx, t - 1, 0] = 0  # the trajectory's Return is 0.
            else:  # receive reward
                fixed_row.append(row_idx)

    return batch[fixed_row]


def _get_obs_component_dim(args):
    move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim = args.obs_component  # [4, (6, 5), (4, 5), 1]
    enemy_feats_dim = np.prod(enemy_feats_dim)
    ally_feats_dim = np.prod(ally_feats_dim)
    return move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim


def _generate_permutation_matrix(bs, seq_length, n_agents, N, device):
    permutation_matrix = th.zeros(size=[bs, seq_length, n_agents, N, N], dtype=th.float32, device=device)
    ordered_list = np.arange(N)  # [0, 1, 2, 3, ...]
    shuffled_list = ordered_list.copy()
    np.random.shuffle(shuffled_list)  # [3, 0, 2, 1, ...]
    permutation_matrix[:, :, :, ordered_list, shuffled_list] = 1
    return permutation_matrix


def do_data_augmentation(args, batch: EpisodeBatch, augment_times=2):
    """
    'obs', 'attack action' and 'available action' need to be transformed
    :param args:
    :param batch:
    :param augment_times:
    :return:
    """
    bs = batch.batch_size
    seq_length = batch.max_seq_length
    obs_component_dim = _get_obs_component_dim(args=args)
    attack_action_start_idx = 6

    augmented_data = []
    for t in range(augment_times):
        new_batch = copy.deepcopy(batch)
        obs = new_batch.data.transition_data["obs"]  # [bs, seq_length, n_agents, obs_dim]
        # actions = new_batch.data.transition_data["actions"]  # [bs, seq_length, n_agents, 1]
        actions_onehot = new_batch.data.transition_data["actions_onehot"]  # [bs, seq_length, n_agents, action_num]
        avail_actions = new_batch.data.transition_data["avail_actions"]  # [bs, seq_length, n_agents, action_num]

        # (1) split observation according to the semantic meaning
        move_feats, enemy_feats, ally_feats, own_feats = th.split(obs, obs_component_dim, dim=-1)
        reshaped_enemy_feats = enemy_feats.contiguous().view(bs, seq_length, args.n_agents, args.n_enemies, -1)
        reshaped_ally_feats = ally_feats.contiguous().view(bs, seq_length, args.n_agents, (args.n_agents - 1), -1)

        # (2) split available action into 2 groups: 'move' and 'attack'.
        avail_other_action = avail_actions[:, :, :, :attack_action_start_idx]  # (no_op, stop, up, down, right, left)
        avail_attack_action = avail_actions[:, :, :, attack_action_start_idx:]  # [n_enemies]

        # (3) split actions_onehot into 2 groups: 'move' and 'attack'.
        other_action_onehot = actions_onehot[:, :, :, :attack_action_start_idx]  # (no_op, stop, up, down, right, left)
        attack_action_onehot = actions_onehot[:, :, :, attack_action_start_idx:]  # [n_enemies]

        # (4) generate permutation matrix for 'ally' and 'enemy'
        ally_perm_matrix = _generate_permutation_matrix(bs, seq_length, args.n_agents, args.n_agents - 1,
                                                        device=obs.device)
        enemy_perm_matrix = _generate_permutation_matrix(bs, seq_length, args.n_agents, args.n_enemies,
                                                         device=obs.device)

        # (5) permute obs: including ally and enemy
        # [bs, seq_length, n_agents, N, N] * [bs, seq_length, n_agents, N, feature_dim]
        permuted_enemy_feat = th.matmul(enemy_perm_matrix, reshaped_enemy_feats).view(bs, seq_length, args.n_agents, -1)
        permuted_ally_feat = th.matmul(ally_perm_matrix, reshaped_ally_feats).view(bs, seq_length, args.n_agents, -1)
        permuted_obs = th.cat([move_feats, permuted_enemy_feat, permuted_ally_feat, own_feats], dim=-1)
        # permuted_obs = th.cat([move_feats, permuted_enemy_feat, ally_feats, own_feats], dim=-1)

        # (6) permute available action (use the same permutation matrix for enemy)
        permuted_avail_attack_action = th.matmul(enemy_perm_matrix, avail_attack_action.unsqueeze(-1).float()).view(
            bs, seq_length, args.n_agents, -1)
        permuted_avail_actions = th.cat([avail_other_action, permuted_avail_attack_action.int()], dim=-1)

        # (7) permute attack_action_onehot (use the same permutation matrix for enemy)
        #     used when obs_last_action is True
        permuted_attack_action_onehot = th.matmul(enemy_perm_matrix, attack_action_onehot.unsqueeze(-1).float()).view(
            bs, seq_length, args.n_agents, -1)
        permuted_action_onehot = th.cat([other_action_onehot, permuted_attack_action_onehot], dim=-1)
        permuted_action = permuted_action_onehot.max(dim=-1, keepdim=True)[1]

        new_batch.data.transition_data["obs"] = permuted_obs
        new_batch.data.transition_data["actions"] = permuted_action
        new_batch.data.transition_data["actions_onehot"] = permuted_action_onehot
        new_batch.data.transition_data["avail_actions"] = permuted_avail_actions

        if augment_times > 1:
            augmented_data.append(new_batch)
    if augment_times > 1:
        return augmented_data
    return new_batch
