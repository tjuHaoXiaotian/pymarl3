import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def get_activation_func(name, hidden_dim):
    """
    'relu'
    'tanh'
    'leaky_relu'
    'elu'
    'prelu'
    :param name:
    :return:
    """
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "tanh":
        return nn.Tanh()
    elif name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    elif name == "elu":
        return nn.ELU(alpha=1., inplace=True)
    elif name == 'prelu':
        return nn.PReLU(num_parameters=hidden_dim, init=0.25)


class Hypernet(nn.Module):
    def __init__(self, input_dim, hidden_dim, main_input_dim, main_output_dim, activation_func, n_heads):
        super(Hypernet, self).__init__()

        self.n_heads = n_heads
        # the output dim of the hypernet
        output_dim = main_input_dim * main_output_dim
        # the output of the hypernet will be reshaped to [main_input_dim, main_output_dim]
        self.main_input_dim = main_input_dim
        self.main_output_dim = main_output_dim

        self.multihead_nn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            get_activation_func(activation_func, hidden_dim),
            nn.Linear(hidden_dim, output_dim * self.n_heads),
        )

    def forward(self, x):
        # [...,  main_output_dim + main_output_dim + ... + main_output_dim]
        # [bs, main_input_dim, n_heads * main_output_dim]
        return self.multihead_nn(x).view([-1, self.main_input_dim, self.main_output_dim * self.n_heads])


class Merger(nn.Module):
    def __init__(self, head, fea_dim):
        super(Merger, self).__init__()
        self.head = head
        if head > 1:
            self.weight = Parameter(th.Tensor(1, head, fea_dim).fill_(1.))
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        :param x: [bs, n_head, fea_dim]
        :return: [bs, fea_dim]
        """
        if self.head > 1:
            return th.sum(self.softmax(self.weight) * x, dim=1, keepdim=False)
        else:
            return th.squeeze(x, dim=1)


class HPN_RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(HPN_RNNAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_allies = args.n_allies
        self.n_enemies = args.n_enemies
        self.n_actions = args.n_actions
        self.n_heads = args.hpn_head_num
        self.rnn_hidden_dim = args.rnn_hidden_dim

        # [4 + 1, (6, 5), (4, 5)]
        self.own_feats_dim, self.enemy_feats_dim, self.ally_feats_dim = input_shape
        self.enemy_feats_dim = self.enemy_feats_dim[-1]  # [n_enemies, feat_dim]
        self.ally_feats_dim = self.ally_feats_dim[-1]  # [n_allies, feat_dim]

        if self.args.obs_agent_id:
            # embedding table for agent_id
            self.agent_id_embedding = th.nn.Embedding(self.n_agents, self.rnn_hidden_dim)

        if self.args.obs_last_action:
            # embedding table for action id
            self.action_id_embedding = th.nn.Embedding(self.n_actions, self.rnn_hidden_dim)

        # Unique Features (do not need hyper net)
        self.fc1_own = nn.Linear(self.own_feats_dim, self.rnn_hidden_dim, bias=True)  # only one bias is OK

        # %%%%%%%%%%%%%%%%%%%%%% Hypernet-based API input layer %%%%%%%%%%%%%%%%%%%%
        # Multiple entities (use hyper net to process these features to ensure permutation invariant)
        self.hyper_input_w_enemy = Hypernet(
            input_dim=self.enemy_feats_dim, hidden_dim=args.hpn_hyper_dim,
            main_input_dim=self.enemy_feats_dim, main_output_dim=self.rnn_hidden_dim,
            activation_func=args.hpn_hyper_activation, n_heads=self.n_heads
        )  # output shape: (enemy_feats_dim * self.rnn_hidden_dim)
        self.hyper_input_w_ally = Hypernet(
            input_dim=self.ally_feats_dim, hidden_dim=args.hpn_hyper_dim,
            main_input_dim=self.ally_feats_dim, main_output_dim=self.rnn_hidden_dim,
            activation_func=args.hpn_hyper_activation, n_heads=self.n_heads
        )  # output shape: ally_feats_dim * rnn_hidden_dim

        # self.unify_input_heads = nn.Linear(self.rnn_hidden_dim * self.n_heads, self.rnn_hidden_dim)
        self.unify_input_heads = Merger(self.n_heads, self.rnn_hidden_dim)

        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)

        self.output_normal_actions = nn.Linear(self.rnn_hidden_dim, args.output_normal_actions)  # (no_op, stop, up, down, right, left)

        # %%%%%%%%%%%%%%%%%%%%%% Hypernet-based APE output layer %%%%%%%%%%%%%%%%%%%%
        # Multiple entities (use hyper net to process these features to ensure permutation invariant)
        self.hyper_output_w_attack_action = Hypernet(
            input_dim=self.enemy_feats_dim, hidden_dim=args.hpn_hyper_dim,
            main_input_dim=self.rnn_hidden_dim, main_output_dim=1,
            activation_func=args.hpn_hyper_activation, n_heads=self.n_heads
        )  # output shape: rnn_hidden_dim * 1
        self.hyper_output_b_attack_action = Hypernet(
            input_dim=self.enemy_feats_dim, hidden_dim=args.hpn_hyper_dim,
            main_input_dim=1, main_output_dim=1,
            activation_func=args.hpn_hyper_activation, n_heads=self.n_heads
        )  # output shape: 1
        # self.unify_output_heads = nn.Linear(self.n_heads, 1)
        self.unify_output_heads = Merger(self.n_heads, 1)

        if self.args.map_type == "MMM":
            assert self.n_enemies >= self.n_agents, "For MMM map, for the reason that the 'attack' and 'rescue' use the same ids in SMAC, n_enemies must >= n_agents"
            self.hyper_output_w_rescue_action = Hypernet(
                input_dim=self.ally_feats_dim, hidden_dim=args.hpn_hyper_dim,
                main_input_dim=self.rnn_hidden_dim, main_output_dim=1,
                activation_func=args.hpn_hyper_activation, n_heads=self.n_heads
            )  # output shape: rnn_hidden_dim * 1
            self.hyper_output_b_rescue_action = Hypernet(
                input_dim=self.ally_feats_dim, hidden_dim=args.hpn_hyper_dim,
                main_input_dim=1, main_output_dim=1,
                activation_func=args.hpn_hyper_activation, n_heads=self.n_heads
            )  # output shape: 1
            self.unify_rescue_output_heads = Merger(self.n_heads, 1)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1_own.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # [bs * n_agents, mv_fea_dim], [bs * n_agents * n_enemies, enemy_fea_dim], [bs * n_agents * n_allies, ally_fea_dim]
        bs, own_feats_t, enemy_feats_t, ally_feats_t, embedding_indices = inputs

        # (1) Own feature
        embedding_own = self.fc1_own(own_feats_t)  # [bs * n_agents, rnn_hidden_dim]

        # (2) ID embeddings
        if self.args.obs_agent_id:
            agent_indices = embedding_indices[0]
            # [bs * n_agents, rnn_hidden_dim * head]
            embedding_own = embedding_own + self.agent_id_embedding(agent_indices).view(
                -1, self.rnn_hidden_dim)
        if self.args.obs_last_action:
            last_action_indices = embedding_indices[-1]
            if last_action_indices is not None:  # t != 0
                # [bs * n_agents, rnn_hidden_dim * head]
                embedding_own = embedding_own + self.action_id_embedding(last_action_indices).view(
                    -1, self.rnn_hidden_dim)

        # (3) Enemy feature: [bs * n_agents * n_enemies, enemy_fea_dim] -> [bs * n_agents * n_enemies, enemy_feats_dim, rnn_hidden_dim * n_heads]
        input_w_enemy = self.hyper_input_w_enemy(enemy_feats_t)
        # [bs * n_agents * n_enemies, 1, enemy_fea_dim] * [bs * n_agents * n_enemies, enemy_fea_dim, rnn_hidden_dim * head] = [bs * n_agents * n_enemies, 1, rnn_hidden_dim * head]
        embedding_enemies = th.matmul(enemy_feats_t.unsqueeze(1), input_w_enemy).view(
            bs * self.n_agents, self.n_enemies, self.n_heads, self.rnn_hidden_dim
        )  # [bs * n_agents, n_enemies, n_head, rnn_hidden_dim]
        embedding_enemies = embedding_enemies.sum(dim=1, keepdim=False)  # [bs * n_agents, n_head, rnn_hidden_dim]

        # (4) Ally features: [bs * n_agents * n_allies, ally_fea_dim] -> [bs * n_agents * n_allies, ally_feats_dim, rnn_hidden_dim * n_heads]
        input_w_ally = self.hyper_input_w_ally(ally_feats_t)
        # [bs * n_agents * n_allies, 1, ally_fea_dim] * [bs * n_agents * n_allies, ally_fea_dim, rnn_hidden_dim * head] = [bs * n_agents * n_allies, 1, rnn_hidden_dim * head]
        embedding_allies = th.matmul(ally_feats_t.unsqueeze(1), input_w_ally).view(
            bs * self.n_agents, self.n_allies, self.n_heads, self.rnn_hidden_dim
        )  # [bs * n_agents, n_allies, n_head, rnn_hidden_dim]
        embedding_allies = embedding_allies.sum(dim=1, keepdim=False)  # [bs * n_agents, n_head, rnn_hidden_dim]
        # Final embedding, merge multiple heads into one. -> [bs * n_agents, n_head, rnn_hidden_dim]
        embedding = embedding_own + self.unify_input_heads(
            embedding_enemies + embedding_allies
        )

        x = F.relu(embedding, inplace=True)
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        hh = self.rnn(x, h_in)  # [bs * n_agents, rnn_hidden_dim]

        # Q-values of normal actions
        q_normal = self.output_normal_actions(hh).view(bs, self.n_agents, -1)  # [bs, n_agents, 6]

        # Q-values of attack actions: [bs * n_agents * n_enemies, enemy_fea_dim] -> [bs * n_agents * n_enemies, rnn_hidden_dim, 1 * n_heads]
        output_w_attack = self.hyper_output_w_attack_action(enemy_feats_t).view(
            bs * self.n_agents, self.n_enemies, self.rnn_hidden_dim, self.n_heads
        ).transpose(1, 2).reshape(  # -> [bs * n_agents, rnn_hidden_dim, n_enemies, n_heads]
            bs * self.n_agents, self.rnn_hidden_dim, self.n_enemies * self.n_heads
        )  # [bs * n_agents, rnn_hidden_dim, n_enemies * heads]
        # b: [bs * n_agents * n_enemies, enemy_fea_dim] -> [bs * n_agents * n_enemies, 1, n_heads]
        output_b_attack = self.hyper_output_b_attack_action(enemy_feats_t).view(
            bs * self.n_agents, self.n_enemies * self.n_heads
        )  # -> [bs * n_agents, n_enemies * head]

        # [bs * n_agents, 1, rnn_hidden_dim] * [bs * n_agents, rnn_hidden_dim, n_enemies * head] = [bs * n_agents, 1, n_enemies * head]
        # -> # [bs * n_agents, n_enemies * head] -> [bs * n_agents * n_enemies, head, 1]
        q_attacks = (th.matmul(hh.unsqueeze(1), output_w_attack).squeeze(1) + output_b_attack).view(-1, self.n_heads,
                                                                                                    1)
        q_attack = self.unify_output_heads(q_attacks).view(  # [bs * n_agents * n_enemies, 1]
            bs, self.n_agents, self.n_enemies
        )  # [bs, n_agents, n_enemies]

        # %%%%%%%%%%%%%%% 'rescue' actions for map_type == "MMM" %%%%%%%%%%%%%%%
        if self.args.map_type == "MMM":
            output_w_rescue = self.hyper_output_w_rescue_action(ally_feats_t).view(
                bs * self.n_agents, self.n_allies, self.rnn_hidden_dim, self.n_heads
            ).transpose(1, 2).reshape(  # -> [bs * n_agents, rnn_hidden_dim, n_allies, n_heads]
                bs * self.n_agents, self.rnn_hidden_dim, self.n_allies * self.n_heads
            )  # [bs * n_agents, rnn_hidden_dim, n_allies * heads]
            # b: [bs * n_agents * n_allies, ally_fea_dim] -> [bs * n_agents * n_allies, 1, n_heads]
            output_b_rescue = self.hyper_output_b_rescue_action(ally_feats_t).view(
                bs * self.n_agents, self.n_allies * self.n_heads
            )  # -> [bs * n_agents, n_allies * head]

            # [bs * n_agents, 1, rnn_hidden_dim] * [bs * n_agents, rnn_hidden_dim, n_allies * head] = [bs * n_agents, 1, n_allies * head]
            # -> # [bs * n_agents, n_allies * head] -> [bs * n_agents * n_allies, head, 1]
            q_rescue = (th.matmul(hh.unsqueeze(1), output_w_rescue).squeeze(1) + output_b_rescue).view(
                -1, self.n_heads, 1)
            q_rescue = self.unify_rescue_output_heads(q_rescue).view(  # [bs * n_agents * n_allies, 1]
                bs, self.n_agents, self.n_allies
            )  # [bs, n_agents, n_allies]
            # For the reason that medivac is the last indexed agent, so the rescue action idx -> [0, n_allies-1]
            right_padding = th.ones_like(q_attack[:, -1:, self.n_allies:], requires_grad=False) * (-9999999)
            modified_q_attack_of_medivac = th.cat([q_rescue[:, -1:, :], right_padding], dim=-1)
            # Merge
            q_attack = th.cat([q_attack[:, :-1], modified_q_attack_of_medivac], dim=1)

        # Concat 2 types of Q-values
        q = th.cat((q_normal, q_attack), dim=-1)  # [bs, n_agents, 6 + n_enemies]
        return q.view(bs, self.n_agents, -1), hh.view(bs, self.n_agents, -1)
