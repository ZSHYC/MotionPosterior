import torch
import torch.nn as nn
from ..builder import HEADS


# FusionLayerTypeA 保持不变...
class FusionLayerTypeA(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        feature_map, attention_map = inputs
        output_1 = feature_map[:, 0, :, :]
        output_2 = feature_map[:, 1, :, :] * attention_map[:, 0, :, :]
        output_3 = feature_map[:, 2, :, :] * attention_map[:, 2, :, :]
        return torch.stack([output_1, output_2, output_3], dim=1)


@HEADS.register_module
class R_STRHead(nn.Module):
    def __init__(self,
                 in_channels=64,
                 out_channels=3,
                 img_size=(288, 512),
                 patch_size=16,
                 embed_dim=256,
                 num_transformer_layers=4,
                 num_transformer_heads=2,
                 IsDraft=False,
                 dropout=True,
                 use_motion_tokens=True,
                 motion_channels=4
                 ):
        super().__init__()
        self.IsDraft = IsDraft
        self.dropout = dropout
        self.fusion_layer = FusionLayerTypeA()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        print("head:", num_transformer_heads)
        # 1. 草稿头 (保持你之前的修改: 1x1 Conv)
        self.draft_head = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False) # V5_loveall 最高为False

        # 2. Patch embedding for draft heatmaps and motion context
        self.embed_conv = nn.Conv2d(
            in_channels=1,  # 坚持只看 1 通道概率图
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        H_img, W_img = img_size
        H_feat, W_feat = H_img // patch_size, W_img // patch_size
        num_patches_per_frame = H_feat * W_feat

        # Positional Embeddings (保持不变)
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, num_patches_per_frame, embed_dim))
        self.time_embed = nn.Parameter(torch.randn(1, 3, embed_dim))
        self.context_dropout = nn.Dropout(p=0.1)
        self.use_motion_tokens = use_motion_tokens
        self.motion_embed_conv = (
            nn.Conv2d(
                motion_channels,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
            )
            if use_motion_tokens else None
        )
        self.motion_embed = (
            nn.Parameter(torch.zeros(1, num_patches_per_frame, embed_dim))
            if use_motion_tokens else None
        )

        # Transformer Encoder (保持不变)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_transformer_heads,
            batch_first=True,
            dim_feedforward=embed_dim * 4,
            dropout=0.1  # 注意：Transformer 内部也有 dropout，保持默认即可
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # 解码头 (保持你之前的修改: PixelShuffle)
        self.decoder_head = nn.Sequential(
            nn.Conv2d(embed_dim, 1 * (patch_size ** 2), kernel_size=1),
            nn.PixelShuffle(patch_size)
        )

        self.final_sigmoid = nn.Sigmoid()

    def forward(self, x, residual_maps):
        # 1. 生成草稿
        draft_logits = self.draft_head(x)

        # 2. 运动融合 (MVDR)
        # draft_clean 是我们最“干净”的草稿，用于最后的残差相加
        draft_clean = self.fusion_layer([draft_logits, residual_maps])

        if self.training and self.dropout == True:
            draft_for_transformer = self.context_dropout(draft_clean)
        else:
            draft_for_transformer = draft_clean

        B, C, H, W = draft_for_transformer.shape
        H_feat, W_feat = H // self.patch_size, W // self.patch_size

        # 3. 时空注意力流程

        # 3.1 拆分 (注意：这里用的是可能被 Dropout 过的 draft_for_transformer)
        draft_prev = draft_for_transformer[:, 0, :, :].unsqueeze(1)
        draft_curr = draft_for_transformer[:, 1, :, :].unsqueeze(1)
        draft_next = draft_for_transformer[:, 2, :, :].unsqueeze(1)

        # 3.2 Patch Embedding
        embd_prev = self.embed_conv(draft_prev)
        embd_curr = self.embed_conv(draft_curr)
        embd_next = self.embed_conv(draft_next)

        # 3.3 展平
        flat_prev = embd_prev.flatten(2).permute(0, 2, 1)  # [B, N, D]
        flat_curr = embd_curr.flatten(2).permute(0, 2, 1)
        flat_next = embd_next.flatten(2).permute(0, 2, 1)


        # 3.4 加位置编码 (保持不变)
        time_prev_embed = self.time_embed[:, 0, :].unsqueeze(1)
        time_curr_embed = self.time_embed[:, 1, :].unsqueeze(1)
        time_next_embed = self.time_embed[:, 2, :].unsqueeze(1)

        in_prev = flat_prev + self.spatial_pos_embed + time_prev_embed
        in_curr = flat_curr + self.spatial_pos_embed + time_curr_embed
        in_next = flat_next + self.spatial_pos_embed + time_next_embed

        # 3.5 拼接 & Transformer
        frame_tokens = torch.cat([in_prev, in_curr, in_next], dim=1)
        final_input_with_pos = frame_tokens
        if self.motion_embed_conv is not None and residual_maps is not None:
            motion_tokens = self.motion_embed_conv(residual_maps)
            motion_tokens = motion_tokens.flatten(2).permute(0, 2, 1)
            motion_tokens = motion_tokens + self.spatial_pos_embed + self.motion_embed
            final_input_with_pos = torch.cat([frame_tokens, motion_tokens], dim=1)
        repaired_sequence = self.transformer_encoder(final_input_with_pos)

        # 3.6 & 3.7 解码回 Logits
        frame_sequence = repaired_sequence[:, :3 * flat_prev.shape[1]]
        repaired_prev_flat, repaired_curr_flat, repaired_next_flat = torch.chunk(frame_sequence, 3, dim=1)

        repaired_prev_feat = repaired_prev_flat.permute(0, 2, 1).reshape(B, self.embed_dim, H_feat, W_feat)
        repaired_curr_feat = repaired_curr_flat.permute(0, 2, 1).reshape(B, self.embed_dim, H_feat, W_feat)
        repaired_next_feat = repaired_next_flat.permute(0, 2, 1).reshape(B, self.embed_dim, H_feat, W_feat)

        out_prev_residual = self.decoder_head(repaired_prev_feat)
        out_curr_residual = self.decoder_head(repaired_curr_feat)
        out_next_residual = self.decoder_head(repaired_next_feat)

        # 4. 残差连接
        residual_logits = torch.cat([out_prev_residual, out_curr_residual, out_next_residual], dim=1)

        final_logits = draft_for_transformer + residual_logits

        # 5. 最终激活
        final_output = self.final_sigmoid(final_logits)

        if self.IsDraft:
            return final_output, self.final_sigmoid(draft_for_transformer)
        else:
            return final_output
