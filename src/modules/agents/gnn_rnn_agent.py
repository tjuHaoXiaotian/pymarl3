import torch as th
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """Implements a GCN layer."""

    def __init__(self, input_dim, output_dim, n_nodes):
        super(GraphConvLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_nodes = n_nodes

        self.lin_layer_neighbor = nn.Linear(input_dim, output_dim)
        self.lin_layer_self = nn.Linear(input_dim, output_dim)

    def forward(self, inputs):
        input_feature, adjacent_matrix = inputs
        # [N, N] * [bs, N, fea_dim]
        neighbors = th.matmul(adjacent_matrix, self.lin_layer_neighbor(input_feature))  # sum aggregation
        neighbors = F.relu(neighbors, inplace=True)

        node_feats = self.lin_layer_self(input_feature)  # node features
        node_feats = F.relu(node_feats, inplace=True)
        out = (node_feats + neighbors) / self.n_nodes  # mean
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'


class GNN(nn.Module):
    """
    A graph net that is used to pre-process input components, and solve the order issue.
    gnn_rnn 35.404K for 5m_vs_6m
    """

    def __init__(self, fea_dim, n_nodes, hidden_size, layer_num=2, out_pool_type='avg'):
        super(GNN, self).__init__()
        self.fea_dim = fea_dim
        self.n_nodes = n_nodes
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.out_pool_type = out_pool_type

        # Adjacent Matrix, assumes a fully connected graph.
        self.register_buffer('adj', (th.ones(n_nodes, n_nodes) - th.eye(n_nodes)))

        # GNNs
        GNN_layers = []
        previous_out_dim = fea_dim
        for _ in range(self.layer_num):
            GNN_layers.append(GraphConvLayer(input_dim=previous_out_dim, output_dim=hidden_size, n_nodes=n_nodes))
            previous_out_dim = hidden_size
        self.gnn_layers = nn.Sequential(*GNN_layers)

    def forward(self, x):
        # GNNs
        out = self.gnn_layers([x, self.adj])

        # Pooling
        if self.out_pool_type == 'avg':
            ret = out.mean(dim=1, keepdim=False)  # Pooling over the node dimension.
        elif self.out_pool_type == 'max':
            ret, _ = out.max(dim=1, keepdim=False)
        else:
            raise NotImplementedError
        return ret


class GnnRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(GnnRNNAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_allies = args.n_allies
        self.n_enemies = args.n_enemies
        self.n_actions = args.n_actions

        # [4 + 1, (6, 5), (4, 5)], take 5m_vs_6m for example
        self.own_feats_dim, self.enemy_feats_dim, self.ally_feats_dim = input_shape
        self.enemy_feats_dim = self.enemy_feats_dim[-1]
        self.ally_feats_dim = self.ally_feats_dim[-1]

        # (1) To transform all kinds of features into the same dimension.
        if self.args.obs_agent_id:
            # embedding table for agent_id
            self.agent_id_embedding = th.nn.Embedding(self.n_agents, self.args.rnn_hidden_dim)
        if self.args.obs_last_action:
            # embedding table for action id
            self.action_id_embedding = th.nn.Embedding(self.n_actions, self.args.rnn_hidden_dim)
        # Own features
        self.fc1_own = nn.Linear(self.own_feats_dim, args.rnn_hidden_dim, bias=True)  # only one bias is OK
        # Ally features
        self.fc1_ally = nn.Linear(self.ally_feats_dim, args.rnn_hidden_dim, bias=False)  # only one bias is OK
        # Enemy features
        self.fc1_enemy = nn.Linear(self.enemy_feats_dim, args.rnn_hidden_dim, bias=False)  # only one bias is OK

        # (2) GNN
        self.gnn = GNN(fea_dim=args.rnn_hidden_dim, n_nodes=self.n_agents + self.n_enemies,
                       hidden_size=args.rnn_hidden_dim, layer_num=args.gnn_layer_num, out_pool_type='avg')

        # (3) RNN and output
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        # print(self.fc1_ally.weight.data.mean(), self.fc1_ally.weight.data.var())
        # print(self.fc1_enemy.weight.data.mean(), self.fc1_enemy.weight.data.var())

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1_own.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state=None):
        # [bs * n_agents, mv_fea_dim+own_fea_dim], [bs * n_agents * n_enemies, enemy_fea_dim], [bs * n_agents * n_allies, ally_fea_dim]
        bs, own_feats_t, enemy_feats_t, ally_feats_t, embedding_indices = inputs

        # %%%%%%%%%% To transform all kinds of features into the same dimension.  %%%%%%%%%%
        # (1) Own feature
        embedding_own = self.fc1_own(own_feats_t)  # [bs * n_agents, rnn_hidden_dim]
        # (2) ID embeddings
        if self.args.obs_agent_id:
            agent_indices = embedding_indices[0]
            # [bs * n_agents, rnn_hidden_dim]
            embedding_own = embedding_own + self.agent_id_embedding(agent_indices).view(-1, self.args.rnn_hidden_dim)
        if self.args.obs_last_action:
            last_action_indices = embedding_indices[-1]
            if last_action_indices is not None:  # t != 0
                # [bs * n_agents, rnn_hidden_dim]
                embedding_own = embedding_own + self.action_id_embedding(last_action_indices).view(
                    -1, self.args.rnn_hidden_dim)
        embedding_own = embedding_own.unsqueeze(dim=1)  # [bs * n_agents, 1, rnn_hidden_dim]
        # (3) Enemy feature
        embedding_enemies = self.fc1_enemy(enemy_feats_t).view(
            bs * self.n_agents, self.n_enemies, self.args.rnn_hidden_dim
        )  # [bs * n_agents, n_enemies, rnn_hidden_dim]
        # (4) Ally features
        embedding_allies = self.fc1_ally(ally_feats_t).view(
            bs * self.n_agents, self.n_allies, self.args.rnn_hidden_dim
        )  # [bs * n_agents, n_allies, rnn_hidden_dim]

        # [bs * n_agents, 1+n_allies+n_enemies, rnn_hidden_dim]
        fea_embeddings = th.cat([embedding_own, embedding_allies, embedding_enemies], dim=1)
        fea_embeddings = F.relu(fea_embeddings, inplace=True)
        # %%%%%%%%%% To transform all kinds of features into the same dimension.  %%%%%%%%%%

        x = self.gnn(fea_embeddings)  # [bs * n_agents, rnn_hidden_dim]

        if hidden_state is not None:
            hidden_state = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, hidden_state)
        q = self.fc2(h)
        return q.view(bs, self.n_agents, -1), h.view(bs, self.n_agents, -1)
