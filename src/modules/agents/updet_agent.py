import torch.nn as nn
import torch.nn.functional as F
import torch
import argparse
import torch as th


class UPDeT(nn.Module):
    def __init__(self, input_shape, args):
        super(UPDeT, self).__init__()
        self.args = args
        self.input_shape = input_shape  # (5, (6, 5), (4, 5)) for 5m vs 6m
        self.n_agents = args.n_agents

        self.transformer = Transformer(input_shapes=input_shape, emb=args.transformer_embed_dim,
                                       heads=args.transformer_heads, depth=args.transformer_depth,
                                       output_dim=args.transformer_embed_dim)
        self.q_basic = nn.Linear(args.transformer_embed_dim, 6)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.q_basic.weight.new(1, self.args.transformer_embed_dim).zero_()

    def forward(self, inputs, hidden_state):
        # (bs * n_agents, 1, transformer_embed_dim]
        hidden_state = hidden_state.reshape(-1, 1, self.args.transformer_embed_dim)

        # transformer-out: torch.Size([b * n_agents, 1+n_enemies+(n_agents-1)+1, transformer_embed_dim])
        # in dim 1: self_fea_att_value, m enemy_fea_att_value, n-1 ally_fea_att_value, hidden_state
        outputs, _ = self.transformer.forward(
            inputs, hidden_state, None)

        # first output for 6 action (no_op stop up down left right)
        q_basic_actions = self.q_basic(outputs[:, 0, :])

        # last dim for hidden state
        h = outputs[:, -1:, :]

        # Replace the loop with batch computing
        # q_enemies_list = []
        # # each enemy has an output Q
        # for i in range(self.args.n_enemies):
        #     q_enemy = self.q_basic(outputs[:, 1 + i, :])
        #     q_enemy_mean = torch.mean(q_enemy, 1, True)
        #     q_enemies_list.append(q_enemy_mean)
        # # concat enemy Q over all enemies
        # q_enemies = torch.stack(q_enemies_list, dim=1).squeeze()

        q_enemies = self.q_basic(
            outputs[:, 1: 1 + self.args.n_enemies, :])  # [bs * n_agents, n_enemies, 32]->[bs * n_agents, n_enemies, 6]
        q_enemies = q_enemies.mean(dim=-1, keepdim=False)  # The average of the Move Action Q

        # concat basic action Q with enemy attack Q
        q = torch.cat((q_basic_actions, q_enemies), 1)

        return q, h  # [bs * n_agents, 6 + n_enemies], this shape will be reshaped to  [bs, n_agents, 6 + n_enemies] in forward() of the BasicMAC


class SelfAttention(nn.Module):
    def __init__(self, emb, heads=8, mask=False):
        super(SelfAttention, self).__init__()

        self.emb = emb
        self.heads = heads
        self.mask = mask

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x, mask):

        b, t, e = x.size()
        h = self.heads
        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b * h, t, t)

        if self.mask:  # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        if mask is not None:
            dot = dot.masked_fill(mask == 0, -1e9)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)  # [b, n_entities, e]


class TransformerBlock(nn.Module):

    def __init__(self, emb, heads, mask, ff_hidden_mult=4, dropout=0.0):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(emb, heads=heads, mask=mask)
        self.mask = mask
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)
        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )
        self.do = nn.Dropout(dropout)

    def forward(self, x_mask):
        x, mask = x_mask
        attended = self.attention(x, mask)
        x = self.norm1(attended + x)
        x = self.do(x)
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)
        x = self.do(x)
        return x, mask


class Transformer(nn.Module):

    def __init__(self, input_shapes, emb, heads, depth, output_dim):
        super(Transformer, self).__init__()
        self.num_tokens = output_dim

        self.input_shapes = input_shapes  # (5, (6, 5), (4, 5)) for 5m vs 6m
        # use the max_dim to init the token layer (to support all maps)
        token_dim = max([input_shapes[0], input_shapes[1][-1], input_shapes[2][-1]])
        self.token_embedding = nn.Linear(token_dim, emb)

        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(emb=emb, heads=heads, mask=False))
        self.tblocks = nn.Sequential(*tblocks)
        self.toprobs = nn.Linear(emb, output_dim)

    def forward(self, inputs, h, mask):
        """

        :param inputs: cat([(bs * n_agents, 1, -1), (bs * n_agents, n_enemies, -1), (bs * n_agents, (n_agents-1), -1)], dim=1)
        :param h:
        :param mask:
        :return:
        """
        tokens = self.token_embedding(inputs) # (bs * n_agents, 1 + n_enemies + (n_agents-1), emb)

        # Append hidden state to the end
        tokens = torch.cat((tokens, h), 1)  # tokens+h: torch.Size([5, 12, 32])

        b, t, e = tokens.size()

        x, mask = self.tblocks((tokens, mask))
        # print("transformer-out:", x.shape)  # transformer-out: torch.Size([5, 12, 32])

        x = self.toprobs(x.view(b * t, e)).view(b, t, self.num_tokens)  # torch.Size([5, 12, 32])
        return x, tokens


def mask_(matrices, maskval=0.0, mask_diagonal=True):
    b, h, w = matrices.size()
    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unit Testing')
    parser.add_argument('--token_dim', default='5', type=int)
    parser.add_argument('--emb', default='32', type=int)
    parser.add_argument('--heads', default='3', type=int)
    parser.add_argument('--depth', default='2', type=int)
    parser.add_argument('--ally_num', default='5', type=int)
    parser.add_argument('--enemy_num', default='5', type=int)
    parser.add_argument('--episode', default='20', type=int)
    args = parser.parse_args()

    # testing the agent
    agent = UPDeT(None, args).cuda()
    hidden_state = agent.init_hidden().cuda().expand(args.ally_num, 1, -1)
    tensor = torch.rand(args.ally_num, args.ally_num + args.enemy_num, args.token_dim).cuda()
    q_list = []
    for _ in range(args.episode):
        q, hidden_state = agent.forward(tensor, hidden_state, args.ally_num, args.enemy_num)
        q_list.append(q)
