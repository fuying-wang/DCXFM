from lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
from einops import rearrange
from dcxfm.third_party.mgca.encoder import ImageEncoder, BertEncoder


class MGCA(LightningModule):
    '''Pytorch lightning implementation of MGCA'''

    def __init__(self,
                 img_encoder: str = "vit_base",
                 freeze_bert: bool = False,
                 emb_dim: int = 128,
                 softmax_temperature: float = 0.07,
                 learning_rate: float = 2e-5,
                 momentum: float = 0.9,
                 weight_decay: float = 0.05,
                 batch_size: int = 64,
                 num_workers: int = 8,
                 # TODO: tune this hyperparameter
                 local_temperature: float = 0.1,
                 proto_temperature: float = 0.2,
                 num_prototypes: int = 500,
                 bidirectional: bool = True,
                 use_local_atten: bool = False,
                 num_heads: int = 1,
                 lamb: float = 0.75,
                 lambda_1: float = 1,
                 lambda_2: float = 0.7,
                 lambda_3: float = 0.5,
                 freeze_prototypes_epochs: int = 1,
                 sinkhorn_iterations: int = 3,
                 epsilon: float = 0.05,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()

        # init encoders
        self.img_encoder_q = ImageEncoder(
            model_name=img_encoder, output_dim=self.hparams.emb_dim)
        self.text_encoder_q = BertEncoder(
            output_dim=self.hparams.emb_dim, freeze_bert=freeze_bert)

        # patch local attention layer
        self.patch_local_atten_layer = nn.MultiheadAttention(
            self.hparams.emb_dim, self.hparams.num_heads, batch_first=True)
        # sentence local attention layer
        self.word_local_atten_layer = nn.MultiheadAttention(
            self.hparams.emb_dim, self.hparams.num_heads, batch_first=True)

        self.prototype_layer = nn.Linear(emb_dim, num_prototypes, bias=False)

        self.tokenizer = self.text_encoder_q.tokenizer

    def encode_image(self, imgs):
        # Forward of query image encoder
        img_feat_q, patch_feat_q = self.img_encoder_q(
            imgs)
        patch_emb_q = self.img_encoder_q.local_embed(patch_feat_q)
        patch_emb_q = F.normalize(patch_emb_q, dim=-1)
        img_emb_q = self.img_encoder_q.global_embed(img_feat_q)
        img_emb_q = F.normalize(img_emb_q, dim=-1)
        return img_emb_q, patch_emb_q

    def encode_text(self, input_ids, attention_mask):
        # Forward of query text encoder
        report_feat_q, word_feat_q, word_attn_q, sents = self.text_encoder_q(
            input_ids, attention_mask)
        word_emb_q = self.text_encoder_q.local_embed(word_feat_q)
        word_emb_q = F.normalize(word_emb_q, dim=-1)
        report_emb_q = self.text_encoder_q.global_embed(report_feat_q)
        report_emb_q = F.normalize(report_emb_q, dim=-1)
        return word_emb_q, report_emb_q

    @torch.no_grad()
    def predict_similarity_map(self, imgs, input_ids, attention_mask):
        img_emb_q, patch_emb_q = self.encode_image(imgs)
        word_emb_q, report_emb_q = self.encode_text(input_ids, attention_mask)

        # if self.hparams.use_local_atten:
        #     word_atten_output, atten_scores = self.word_local_atten_layer(
        #         word_emb_q, patch_emb_q, patch_emb_q)
        # else:
        #     atten_scores = torch.bmm(word_emb_q, patch_emb_q.permute(0, 2, 1))
        #     # word_num = word_emb_q.size(1)
        #     # atten_scores = F.softmax(
        #     #     atten_sim / self.hparams.local_temperature, dim=-1)  # bz, 196, 111
        #     # word_atten_output = torch.bmm(atten_scores, patch_emb_q)

        # # average the attention scores
        # similarity_map = torch.mean(atten_scores, dim=1)
        # h = w = int(similarity_map.size(1)**0.5)
        # similarity_map = rearrange(
        #     similarity_map, 'b (h w) -> b h w', h=h, w=w).unsqueeze(0)
        
        similarity_map = torch.bmm(
            report_emb_q.unsqueeze(1), patch_emb_q.permute(0, 2, 1))
        h = w = int(similarity_map.size(-1)**0.5)
        similarity_map = rearrange(
            similarity_map, 'b c (h w) -> b c h w', h=h, w=w)

        return similarity_map


if __name__ == "__main__":
    ckpt_path = "/home/fywang/Documents/CXRSeg/pretrained/MGCA/resnet_50.ckpt"
    model = MGCA.load_from_checkpoint(ckpt_path)
    print(model.hparams)
