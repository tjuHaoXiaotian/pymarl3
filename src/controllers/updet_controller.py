from .basic_controller import BasicMAC
import torch as th
import numpy as np
import torch.nn.functional as F


class UPDETController(BasicMAC):
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        if t_ep == 0:
            self.set_evaluation_mode()
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
                                                            test_mode=test_mode)
        return chosen_actions

    def _get_obs_shape(self):
        size = 0
        for comp in self.args.obs_component:
            if isinstance(comp, int):
                size += comp
            else:
                size += np.prod(comp)
        return size

    def _get_obs_component_dim(self):
        move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim = self.args.obs_component  # [4, (6, 5), (4, 5), 1]
        enemy_feats_dim = np.prod(enemy_feats_dim)
        ally_feats_dim = np.prod(ally_feats_dim)
        return move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        raw_obs = batch["obs"][:, t]  # [batch, agent_num, obs_dim]
        # assert raw_obs.shape[-1] == self._get_obs_shape()
        obs_component_dim = self._get_obs_component_dim()
        move_feats, enemy_feats, ally_feats, own_feats = th.split(raw_obs, obs_component_dim, dim=-1)
        own_context = th.cat((own_feats, move_feats), dim=2)
        # use the max_dim (over self, enemy and ally) to init the token layer (to support all maps)
        token_dim = max([self.input_shape[0], self.input_shape[1][-1], self.input_shape[2][-1]])

        own_context = own_context.contiguous().view(bs * self.n_agents, 1, -1)
        enemy_feats = enemy_feats.contiguous().view(bs * self.n_agents, self.args.n_enemies, -1)
        ally_feats = ally_feats.contiguous().view(bs * self.n_agents, (self.args.n_agents - 1), -1)

        # In the original repository, UPDeT only supports marine-based battle scenarios. e.g. 3m, 8m, 5m_vs_6m, whose feature_dim is the same
        # We do zero paddings here to support all maps
        inputs = th.cat([
            self.zero_padding(own_context, token_dim),
            self.zero_padding(enemy_feats, token_dim),
            self.zero_padding(ally_feats, token_dim)
        ], dim=1)

        return inputs

    def _get_input_shape(self, scheme):
        move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim = self.args.obs_component
        own_context_dim = move_feats_dim + own_feats_dim
        return own_context_dim, enemy_feats_dim, ally_feats_dim

    def zero_padding(self, features, token_dim):
        """
        :param features: [bs * n_agents, k, fea_dim]
        :param token_dim: maximum of fea_dim
        :return:
        """
        existing_dim = features.shape[-1]
        if existing_dim < token_dim:
            # padding to the right side of the last dimension of the feature.
            return F.pad(features, pad=[0, token_dim - existing_dim], mode='constant', value=0)
        else:
            return features
