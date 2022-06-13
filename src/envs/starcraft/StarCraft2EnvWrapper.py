#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：API-Network 
@File    ：StarCraft2EnvWrapper.py
@Author  ：Hao Xiaotian
@Date    ：2022/6/13 16:26 
'''

from smac.env.starcraft2.starcraft2 import StarCraft2Env


class StarCraft2EnvWrapper(StarCraft2Env):

    # Add new functions to support permutation operation
    def get_obs_component(self):
        move_feats_dim = self.get_obs_move_feats_size()
        enemy_feats_dim = self.get_obs_enemy_feats_size()
        ally_feats_dim = self.get_obs_ally_feats_size()
        own_feats_dim = self.get_obs_own_feats_size()
        obs_component = [move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim]
        return obs_component

    def get_state_component(self):
        if self.obs_instead_of_state:
            return [self.get_obs_size()] * self.n_agents

        nf_al = 4 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits

        enemy_state = self.n_enemies * nf_en
        ally_state = self.n_agents * nf_al

        size = [ally_state, enemy_state]

        if self.state_last_action:
            size.append(self.n_agents * self.n_actions)
        if self.state_timestep_number:
            size.append(1)
        return size

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "n_enemies": self.n_enemies,
            "episode_limit": self.episode_limit,
            "obs_ally_feats_size": self.get_obs_ally_feats_size(),
            "obs_enemy_feats_size": self.get_obs_enemy_feats_size(),
            "state_ally_feats_size": 4 + self.shield_bits_ally + self.unit_type_bits,
            "state_enemy_feats_size": 3 + self.shield_bits_enemy + self.unit_type_bits,
            "obs_component": self.get_obs_component(),
            "state_component": self.get_state_component(),
            "map_type": self.map_type,
        }
        print(env_info)
        return env_info

    def _get_medivac_ids(self):
        medivac_ids = []
        for al_id, al_unit in self.agents.items():
            if self.map_type == "MMM" and al_unit.unit_type == self.medivac_id:
                medivac_ids.append(al_id)
        print(medivac_ids)  # [9]
        return medivac_ids
