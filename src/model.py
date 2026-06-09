import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GraphConv
from torch.nn.modules.batchnorm import _BatchNorm

from src.utils import CustomPiecewiseActivation


class NodeLevelBatchNorm(_BatchNorm):
    """Node-level batch normalization."""
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError(f'expected 2D input (got {input.dim()}D input)')

    def forward(self, input):
        self._check_input_dim(input)
        exponential_average_factor = 0.0 if self.momentum is None else self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)

        return F.batch_norm(
            input, self.running_mean, self.running_var,
            self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps
        )


class GraphConvBn(nn.Module):
    """Graph convolution with batch normalization and activation."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GraphConv(in_channels, out_channels)
        self.norm = NodeLevelBatchNorm(out_channels)
        self.activation = CustomPiecewiseActivation()

    def forward(self, x, edge_index):
        out = self.conv(x, edge_index)
        out = self.norm(out)
        out = self.activation(out)
        return out


class DenseLayer(nn.Module):
    """Dense layer for graph neural network."""
    
    def __init__(self, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        self.conv1 = GraphConvBn(num_input_features, growth_rate * bn_size)
        self.conv2 = GraphConvBn(growth_rate * bn_size, growth_rate)

    def forward(self, x, edge_index):
        concated = x
        out = self.conv1(concated, edge_index)
        out = self.conv2(out, edge_index)
        return out


class DenseBlock(nn.Module):
    """Dense block for graph neural network."""
    
    def __init__(self, num_layers, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate, bn_size
            )
            self.layers.append(layer)

    def forward(self, x, edge_index):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1), edge_index)
            features.append(out)
        return torch.cat(features, dim=1)


class TransitionLayer(nn.Module):
    """Transition layer for graph neural network."""
    
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.conv = GraphConvBn(num_input_features, num_output_features)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class MGNN(nn.Module):
    """Multi-scale Graph Neural Network for compound encoding."""
    
    def __init__(self, num_input_features=87, out_dim=256, growth_rate=32,
                 block_config=[8, 8, 8], bn_sizes=[2, 2, 2]):
        super().__init__()

        self.initial_conv = GraphConvBn(num_input_features, 32)

        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        num_features = 32
        for i, (num_layers, bn_size) in enumerate(zip(block_config, bn_sizes)):
            block = DenseBlock(num_layers, num_features, growth_rate, bn_size)
            self.blocks.append(block)
            num_features += num_layers * growth_rate

            if i < len(block_config) - 1:
                trans = TransitionLayer(num_features, num_features // 2)
                self.transitions.append(trans)
                num_features = num_features // 2

        self.projection = nn.Linear(num_features, out_dim)
        self.out_dim = out_dim

    def forward(self, x, edge_index, batch):
        x = self.initial_conv(x, edge_index)

        for i, block in enumerate(self.blocks):
            x = block(x, edge_index)
            if i < len(self.transitions):
                x = self.transitions[i](x, edge_index)

        node_features = x
        graph_features = global_mean_pool(node_features, batch)
        graph_features = self.projection(graph_features)

        return node_features, graph_features


class BidirectionalCrossAttention(nn.Module):
    """Bidirectional Cross-Attention module."""
    
    def __init__(self, embed_dim=640, num_heads=8, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.W_Q_C = nn.Linear(embed_dim, embed_dim)
        self.W_K_P = nn.Linear(embed_dim, embed_dim)
        self.W_V_P = nn.Linear(embed_dim, embed_dim)

        self.W_Q_P = nn.Linear(embed_dim, embed_dim)
        self.W_K_C = nn.Linear(embed_dim, embed_dim)
        self.W_V_C = nn.Linear(embed_dim, embed_dim)

        self.out_proj_C = nn.Linear(embed_dim, embed_dim)
        self.out_proj_P = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, compound_features, protein_features):
        batch_size, n_atoms, _ = compound_features.shape
        _, n_residues, _ = protein_features.shape

        # Compound to Protein Attention
        Q_C = self.W_Q_C(compound_features)
        K_P = self.W_K_P(protein_features)
        V_P = self.W_V_P(protein_features)

        Q_C = Q_C.view(batch_size, n_atoms, self.num_heads, self.head_dim).transpose(1, 2)
        K_P = K_P.view(batch_size, n_residues, self.num_heads, self.head_dim).transpose(1, 2)
        V_P = V_P.view(batch_size, n_residues, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores_C2P = torch.matmul(Q_C, K_P.transpose(-2, -1)) * self.scale
        attn_weights_C2P = F.softmax(attn_scores_C2P, dim=-1)
        attn_weights_C2P = self.dropout(attn_weights_C2P)

        attn_out_C = torch.matmul(attn_weights_C2P, V_P)
        attn_out_C = attn_out_C.transpose(1, 2).contiguous().view(batch_size, n_atoms, self.embed_dim)
        updated_compound = self.out_proj_C(attn_out_C)

        # Protein to Compound Attention
        Q_P = self.W_Q_P(protein_features)
        K_C = self.W_K_C(compound_features)
        V_C = self.W_V_C(compound_features)

        Q_P = Q_P.view(batch_size, n_residues, self.num_heads, self.head_dim).transpose(1, 2)
        K_C = K_C.view(batch_size, n_atoms, self.num_heads, self.head_dim).transpose(1, 2)
        V_C = V_C.view(batch_size, n_atoms, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores_P2C = torch.matmul(Q_P, K_C.transpose(-2, -1)) * self.scale
        attn_weights_P2C = F.softmax(attn_scores_P2C, dim=-1)
        attn_weights_P2C = self.dropout(attn_weights_P2C)

        attn_out_P = torch.matmul(attn_weights_P2C, V_C)
        attn_out_P = attn_out_P.transpose(1, 2).contiguous().view(batch_size, n_residues, self.embed_dim)
        updated_protein = self.out_proj_P(attn_out_P)

        updated_compound = updated_compound + compound_features
        updated_protein = updated_protein + protein_features

        return updated_compound, updated_protein, attn_weights_C2P, attn_weights_P2C


class CrossAttnDeepHybridCPI(nn.Module):
    """Main model class for Compound-Protein Interaction prediction."""
    
    def __init__(self,
                 compound_input_dim=87,
                 compound_out_dim=640,
                 protein_embed_dim=640,
                 cross_attn_heads=8,
                 cross_attn_dropout=0.1,
                 num_classes=2):
        super().__init__()

        self.mgnn = MGNN(
            num_input_features=compound_input_dim,
            out_dim=compound_out_dim,
            growth_rate=32,
            block_config=[8, 8, 8]
        )

        self.project_compound_nodes = nn.Linear(456, protein_embed_dim)

        self.cross_attention = BidirectionalCrossAttention(
            embed_dim=protein_embed_dim,
            num_heads=cross_attn_heads,
            dropout=cross_attn_dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(protein_embed_dim * 2, 1024),
            CustomPiecewiseActivation(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            CustomPiecewiseActivation(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            CustomPiecewiseActivation(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, compound_x, compound_edge_index, compound_batch,
                protein_embeddings, return_attention=False):

        batch_size = protein_embeddings.shape[0]

        node_features, graph_features = self.mgnn(
            compound_x, compound_edge_index, compound_batch
        )

        node_features_projected = self.project_compound_nodes(node_features)

        compound_node_list = []
        for i in range(batch_size):
            mask = (compound_batch == i)
            compound_node_list.append(node_features_projected[mask])

        max_atoms = max([len(nodes) for nodes in compound_node_list])
        padded_compound_nodes = torch.zeros(batch_size, max_atoms, node_features_projected.shape[-1])
        padded_compound_nodes = padded_compound_nodes.to(compound_x.device)

        for i, nodes in enumerate(compound_node_list):
            padded_compound_nodes[i, :len(nodes), :] = nodes

        protein_embeddings = protein_embeddings.unsqueeze(1)

        updated_compound, updated_protein, attn_C2P, attn_P2C = self.cross_attention(
            padded_compound_nodes, protein_embeddings
        )

        compound_pooled = updated_compound.mean(dim=1)
        protein_pooled = updated_protein.mean(dim=1)

        combined = torch.cat([compound_pooled, protein_pooled], dim=1)
        logits = self.classifier(combined)

        if return_attention:
            return logits, (attn_C2P, attn_P2C)

        return logits
