import torch
import torch.nn as nn
import MinkowskiEngine as ME
import vision_transformer as vits
from vot.models.factory import create_model
import yaml


class Hybrid(nn.Module):
    def __init__(self, dino=None, vot=None):
        super().__init__()
        if dino is None:
            dino = vits.__dict__["vit_small"](patch_size=16)
        if vot is None:
            with open(
                "/home/hli/dl_ws/voxel-transformer/configs/2d_dino.yaml", "r"
            ) as file:
                config = yaml.load(file, Loader=yaml.FullLoader)
            vot = create_model(config["model"]).backbone
        self.dino = dino
        self.vot = vot

    def forward(self, x, debug=False, class_token_only=True):
        ### dino ###
        # x, patch_emb, pos_emb = self.prepare_tokens(x)
        # masks = torch.zeros(x.size(0), x.size(1), dtype=bool, device=x.device)
        # hidden_states = []
        # for blk in self.dino.blocks:
        #     x = blk(x)
        #     hidden_states.append(x)
        # x = self.dino.norm(x)

        ### vot ###
        x = ME.to_sparse(x)
        patch_emb, patch_ids, sparse_patch_embeddings = self.vot.patch_embedding(x)
        pos_emb = self.vot.position_embedding(patch_ids)
        x = patch_emb + pos_emb
        x, masks = self.vot.class_token(x, patch_ids < 0)

        x, hidden_states = self.vot.encoder(x, masks)

        if debug:
            return x, patch_emb, pos_emb, hidden_states
        if class_token_only:
            return x[:, 0]
        return x[:, 1:]

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        sx = ME.to_sparse(x)
        patch_emb, patch_ids, _ = self.vot.patch_embedding(sx)
        # x = torch.zeros_like(patch_emb)
        # for b in range(B):
        #     x[b, patch_ids[b]] = patch_emb[b]
        # patch_emb = x
        x = patch_emb

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.dino.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        # pos_emb = self.dino.pos_embed
        # pos_emb = self.vot.position_embedding.position_embedding.unsqueeze(0)
        pos_emb = self.vot.position_embedding(patch_ids)
        cls_emb = self.dino.pos_embed[:, 0:1]
        pos_emb = torch.cat((cls_emb.expand(B, -1, -1), pos_emb), 1)

        x = x + pos_emb
        # pos_emb = self.dino.interpolate_pos_encoding(x, w, h)

        return self.dino.pos_drop(x), patch_emb, pos_emb
