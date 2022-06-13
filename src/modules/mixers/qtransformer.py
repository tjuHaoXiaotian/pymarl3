import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class HyperLinear(nn.Module):
    def __init__(self, entity_num, input_dim, output_dim, hyper_hidden_size, bias=True):
        super(HyperLinear, self).__init__()
        self.normalize = False
        self.entity_num = entity_num
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.hypernet = nn.Sequential(
            nn.Linear(input_dim, hyper_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hyper_hidden_size, input_dim * output_dim),
            # nn.Tanh()
        )
        if bias:
            self.bias = Parameter(th.Tensor(1, output_dim).fill_(0.))
        else:
            self.bias = 0

    def forward(self, x):
        bs, fea_dim = x.shape
        hyper_out = self.hypernet(x)

        if self.normalize:
            # [batch_size, input_dim * output_dim] -> [b * t, entity_num, input_dim, output_dim]
            hyper_out = F.softmax(hyper_out.view(-1, self.entity_num, self.input_dim, self.output_dim), dim=1)

        # [batch_size, input_dim * output_dim] -> [batch_size, input_dim, output_dim]
        weights = hyper_out.view(bs, self.input_dim, self.output_dim)

        out = th.matmul(x.unsqueeze(1), weights).squeeze(1) + self.bias
        return out  # [batch_size output_dim]


class APIEmbeddingLayer(nn.Module):
    def __init__(self, args, output_dim):
        super(APIEmbeddingLayer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_enemies = args.n_enemies
        self.n_actions = args.n_actions
        self.output_dim = output_dim

        self.embedding_enemy = HyperLinear(args.n_enemies, args.state_enemy_feats_size, output_dim, args.hpn_hyper_dim)
        self.embedding_ally = HyperLinear(args.n_agents, args.state_ally_feats_size, output_dim, args.hpn_hyper_dim)

        if self.args.env_args["state_last_action"]:
            self.embedding_action = nn.Linear(args.n_actions, output_dim)

        if self.args.env_args["state_timestep_number"]:
            self.embedding_timestep = nn.Linear(1, output_dim)

    def forward(self, state_components):
        ally_features, enemy_features = state_components[:2]
        ally_features = ally_features.reshape(-1, self.args.state_ally_feats_size)
        enemy_features = enemy_features.reshape(-1, self.args.state_enemy_feats_size)

        # [bs * t, output_dim]
        embed_ally = self.embedding_ally(ally_features).view(-1, self.n_agents, self.output_dim).mean(dim=1)
        embed_enemy = self.embedding_enemy(enemy_features).view(-1, self.n_enemies, self.output_dim).mean(dim=1)
        output = embed_ally + embed_enemy

        if self.args.env_args["state_last_action"]:
            n_agent_actions = state_components[2].reshape(-1, self.n_agents, self.n_actions)
            embed_last_action = self.embedding_action(n_agent_actions).mean(dim=1)  # [bs * t,output_dim]
            output = output + embed_last_action

        if self.args.env_args["state_timestep_number"]:
            timestep = state_components[-1]
            embed_timestep = self.embedding_timestep(timestep)  # [bs * t, output_dim]
            output = output + embed_timestep

        return output


class APIMixer(nn.Module):
    """
    The Mixing Net should be permutation invariant.
    """

    def __init__(self, args):
        super(APIMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.embed_dim = args.mixing_embed_dim
        self.input_dim = self.state_dim = int(np.prod(args.state_shape))

        # hyper w1
        self.hyper_w1 = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, 1 * self.embed_dim)
        )

        # shared PI state embedding
        self.shared_state_embedding = nn.Sequential(
            APIEmbeddingLayer(args, args.hypernet_embed),
            nn.ReLU(inplace=True),
        )

        # hyper b1
        self.hyper_b1 = nn.Sequential(
            nn.Linear(args.hypernet_embed, self.embed_dim)
        )

        # hyper w2 b2
        self.hyper_w2 = nn.Sequential(
            nn.Linear(args.hypernet_embed, self.embed_dim)
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(args.hypernet_embed, 1)
        )

    def forward(self, qvals, states, hidden_states):
        """
        :param qvals: individual Q
        :param states: global state
        :param hidden_states: GRU output of the agent network, [bs, traj_len, n_agents, hidden_dim]
        :return:
        """
        # reshape
        b, t, _ = qvals.size()

        qvals = qvals.reshape(b * t, 1, self.n_agents)
        states = states.reshape(-1, self.state_dim)
        state_components = th.split(states, self.args.state_component, dim=-1)

        # Shared state embedding
        state_embedding = self.shared_state_embedding(state_components)  # [bs * t, hypernet_embed]

        # First layer
        w1 = self.hyper_w1(hidden_states).view(-1, self.n_agents, self.embed_dim)  # [b * t, n_agents, emb]
        w1 = F.softmax(w1, dim=1)  # already be positive
        # [b * t, 1, n_agents] * [b * t, n_agents, emb]

        b1 = self.hyper_b1(state_embedding).view(-1, 1, self.embed_dim)

        # Second layer
        w2 = self.hyper_w2(state_embedding).view(-1, self.embed_dim, 1)  # [b * t, emb, 1]
        b2 = self.hyper_b2(state_embedding).view(-1, 1, 1)  # [b * t, 1, 1]

        # positive weight
        # w1 = th.abs(w1)
        w2 = th.abs(w2)
        # print(w1.mean(), w1.var())
        # print(w2.mean(), w2.var())

        # Forward
        hidden = F.elu(th.matmul(qvals, w1) + b1)  # [b * t, 1, emb]
        y = th.matmul(hidden, w2) + b2  # b * t, 1, 1

        return y.view(b, t, -1)
