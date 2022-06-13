import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class TokenLayer(nn.Module):
    def __init__(self, args, token_dim):
        super(TokenLayer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_enemies = args.n_enemies
        self.n_actions = args.n_actions

        self.embedding_ally = nn.Linear(args.state_ally_feats_size, token_dim)
        self.embedding_enemy = nn.Linear(args.state_enemy_feats_size, token_dim)

        if self.args.env_args["state_last_action"]:
            self.embedding_action = nn.Linear(args.n_actions, token_dim)

        if self.args.env_args["state_timestep_number"]:
            self.embedding_timestep = nn.Linear(1, token_dim)

    def forward(self, state_components):
        ally_features, enemy_features = state_components[:2]
        ally_features = ally_features.reshape(-1, self.n_agents, self.args.state_ally_feats_size)
        enemy_features = enemy_features.reshape(-1, self.n_enemies, self.args.state_enemy_feats_size)

        embed_ally = self.embedding_ally(ally_features)  # [bs * t, n_agents, embed_dim]
        embed_enemy = self.embedding_enemy(enemy_features)  # [bs * t, n_enemies, embed_dim]
        tokens = [embed_ally, embed_enemy]

        if self.args.env_args["state_last_action"]:
            n_agent_actions = state_components[2].reshape(-1, self.n_agents, self.n_actions)
            embed_last_action = self.embedding_action(n_agent_actions)  # [bs * t, n_agents, embed_dim]
            tokens.append(embed_last_action)

        if self.args.env_args["state_timestep_number"]:
            timestep = state_components[-1]
            embed_timestep = self.embedding_timestep(timestep).unsqueeze(dim=-2)  # [bs * t, 1, embed_dim]
            tokens.append(embed_timestep)

        tokens = th.cat(tokens, dim=-2)
        return tokens  # [bs * t, entity_num, embed_dim]


class SelfAttention(nn.Module):
    def __init__(self, emb_dim, shared_query, end_index, heads=1):
        super(SelfAttention, self).__init__()

        self.emb_dim = emb_dim
        self.heads = heads
        self.shared_query = shared_query
        self.end_index = end_index

        if shared_query:
            self.queries = Parameter(th.Tensor(1, emb_dim * heads))
            nn.init.normal_(self.queries)
            self.end_index = 1
        else:
            self.toqueries = nn.Linear(emb_dim, emb_dim * heads, bias=False)
        self.tokeys = nn.Linear(emb_dim, emb_dim * heads, bias=False)
        self.tovalues = nn.Linear(emb_dim, emb_dim * heads, bias=False)

        if self.heads > 1:
            self.unifyheads = nn.Linear(heads * emb_dim, emb_dim)

    def forward(self, x):
        b, t, e = x.size()  # [bs, sequence_length, token_dim]
        h = self.heads

        if self.shared_query:
            queries = self.queries.expand(b, -1).view(b, 1, h, e)
        else:
            queries = self.toqueries(x).view(b, t, h, e)
        keys = self.tokeys(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        if self.shared_query:
            queries = queries.transpose(1, 2).contiguous().view(b * h, 1, e)
        else:
            queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        # This should be more memory efficient
        queries = queries[:, :self.end_index] / (e ** (1 / 4))  # [b * h, entity_num, e]
        keys = keys / (e ** (1 / 4))  # [b * h, t, e]

        # - get dot product of queries and keys, and scale
        dot = th.bmm(queries, keys.transpose(1, 2))  # [b * h, entity_num, t]

        assert dot.size() == (b * h, self.end_index, t)

        # - dot now has row-wise self-attention probabilities
        dot = F.softmax(dot, dim=2)  # [b * h, entity_num, t]

        # apply the self attention to the values, [b * h, entity_num, t] * [b * h, t, token_dim] = [b * h, entity_num, token_dim]
        out = th.bmm(dot, values).view(b, h, self.end_index, e)  # [b, h, entity_num, token_dim]

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, self.end_index, h * e)

        if self.heads > 1:
            return self.unifyheads(out)  # [b, entity_num, token_dim]
        else:
            return out  # [b, entity_num, token_dim]


class TransformerMixer(nn.Module):
    """
    The Mixing Net should be permutation invariant.
    """

    def __init__(self, args):
        super(TransformerMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.embed_dim = args.mixing_embed_dim
        self.input_dim = self.state_dim = int(np.prod(args.state_shape))

        # hyper w1 b1
        self.hyper_w1 = nn.Sequential(
            TokenLayer(args, args.hypernet_embed),
            SelfAttention(args.hypernet_embed, shared_query=False, end_index=self.n_agents, heads=1),
            nn.Linear(args.hypernet_embed, self.embed_dim)
        )
        self.hyper_b1 = nn.Sequential(
            TokenLayer(args, self.embed_dim),
            SelfAttention(self.embed_dim, shared_query=True, end_index=1, heads=1)
        )

        # hyper w2 b2
        self.hyper_w2 = nn.Sequential(
            TokenLayer(args, args.hypernet_embed),
            SelfAttention(args.hypernet_embed, shared_query=True, end_index=1, heads=1),
            nn.Linear(args.hypernet_embed, self.embed_dim)
        )
        self.hyper_b2 = nn.Sequential(
            TokenLayer(args, self.embed_dim),
            SelfAttention(self.embed_dim, shared_query=True, end_index=1, heads=1),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, qvals, states):
        """
        :param qvals: individual Q
        :param states: global state
        :return:
        """
        # reshape
        b, t, _ = qvals.size()

        qvals = qvals.view(b * t, 1, self.n_agents)
        states = states.reshape(-1, self.state_dim)
        state_components = th.split(states, self.args.state_component, dim=-1)

        # First layer
        w1 = self.hyper_w1(state_components).view(-1, self.n_agents, self.embed_dim)  # [b * t, n_agents, emb]
        b1 = self.hyper_b1(state_components).view(-1, 1, self.embed_dim)

        # Second layer
        w2 = self.hyper_w2(state_components).view(-1, self.embed_dim, 1)  # b * t, emb, 1
        b2 = self.hyper_b2(state_components).view(-1, 1, 1)

        # positive weight
        w1 = th.abs(w1)
        w2 = th.abs(w2)
        # print(w1.mean(), w1.var())
        # print(w2.mean(), w2.var())

        # Forward
        hidden = F.elu(th.matmul(qvals, w1) + b1)  # [b * t, 1, emb]
        y = th.matmul(hidden, w2) + b2  # b * t, 1, 1

        return y.view(b, t, -1)
