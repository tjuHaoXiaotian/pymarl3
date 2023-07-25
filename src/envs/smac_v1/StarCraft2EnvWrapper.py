#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：API-Network 
@File    ：StarCraft2EnvWrapper.py
@Author  ：Hao Xiaotian
@Date    ：2022/6/13 16:26 
'''

from .official.starcraft2 import StarCraft2Env

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

            "n_normal_actions": self.n_actions_no_attack,
            "n_allies": self.n_agents - 1,
            # "obs_ally_feats_size": self.get_obs_ally_feats_size(),
            # "obs_enemy_feats_size": self.get_obs_enemy_feats_size(),
            "state_ally_feats_size": self.get_ally_num_attributes(),  # 4 + self.shield_bits_ally + self.unit_type_bits,
            "state_enemy_feats_size": self.get_enemy_num_attributes(),
            # 3 + self.shield_bits_enemy + self.unit_type_bits,
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

    # def reward_battle(self):
    #     """Reward function when self.reward_spare==False.
    #
    #     Fix the **REWARD FUNCTION BUG** of the original starcraft2.py.
    #
    #     We carefully check the code and indeed find some code error in starcraft2.py.
    #     The error is caused by the incorrect reward calculation for the shield regeneration process and this error will
    #     only occur for scenarios where the enemies are Protoss units.
    #
    #     (1) At line 717 of reward_battle() of starcraft2.py, the reward is computed as: reward = abs(delta_enemy).
    #         Normally, when the agents attack the enemies, delta_enemy will > 0 and thus the agents will be rewarded for attacking enemies.
    #
    #     (2) For Protoss enemies, delta_enemy can < 0 due to the shield regeneration. However, due to the abs() taken over delta_enemy,
    #         the agents will still be rewarded when the enemies' shields regenerate. This incorrect reward will lead to undesired behaviors,
    #         e.g., attacking the enemies but not killing them and waiting their shields regenerating.
    #
    #     (3) Due to the PI/PE design and the improved representational capacity, HPN-QMIX is more sensitive to such
    #         incorrect rewards and sometimes learn strange behaviors.
    #
    #     Returns accumulative hit/shield point damage dealt to the enemy
    #     + reward_death_value per enemy unit killed, and, in case
    #     self.reward_only_positive == False, - (damage dealt to ally units
    #     + reward_death_value per ally unit killed) * self.reward_negative_scale
    #     """
    #     if self.reward_sparse:
    #         return 0
    #
    #     reward = 0
    #     delta_deaths = 0
    #     delta_ally = 0
    #     delta_enemy = 0
    #
    #     neg_scale = self.reward_negative_scale
    #
    #     # update deaths
    #     for al_id, al_unit in self.agents.items():
    #         if not self.death_tracker_ally[al_id]:
    #             # did not die so far
    #             prev_health = (
    #                     self.previous_ally_units[al_id].health
    #                     + self.previous_ally_units[al_id].shield
    #             )
    #             if al_unit.health == 0:
    #                 # just died
    #                 self.death_tracker_ally[al_id] = 1
    #                 if not self.reward_only_positive:
    #                     delta_deaths -= self.reward_death_value * neg_scale
    #                 delta_ally += prev_health * neg_scale
    #             else:
    #                 # still alive
    #                 delta_ally += neg_scale * (
    #                         prev_health - al_unit.health - al_unit.shield
    #                 )
    #
    #     for e_id, e_unit in self.enemies.items():
    #         if not self.death_tracker_enemy[e_id]:
    #             prev_health = (
    #                     self.previous_enemy_units[e_id].health
    #                     + self.previous_enemy_units[e_id].shield
    #             )
    #             if e_unit.health == 0:
    #                 self.death_tracker_enemy[e_id] = 1
    #                 delta_deaths += self.reward_death_value
    #                 delta_enemy += prev_health
    #             else:
    #                 delta_enemy += prev_health - e_unit.health - e_unit.shield
    #
    #     if self.reward_only_positive:
    #         ###### reward = abs(delta_enemy + delta_deaths)  # shield regeneration (the original wrong implementation)
    #         # reward = max(delta_enemy, 0) + delta_deaths  # only consider the shield damage
    #         reward = delta_enemy + delta_deaths  # consider the `+shield-damage` and the `-shield-regeneration`
    #     else:
    #         reward = delta_enemy + delta_deaths - delta_ally
    #
    #     return reward