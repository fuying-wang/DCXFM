'''
From https://github.com/marshuang80/gloria/blob/main/gloria/models/vision_model.py
This script defines the modules of GLoRIA
'''
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from einops import rearrange
from .cnn_backbones import resnet_50
from .vits import create_vit

import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    def __init__(self, model_name="resnet50", embedding_dim=768, pretrained=True, freeze_cnn=False):
        super(ImageEncoder, self).__init__()

        self.output_dim = embedding_dim
        self.model_name = model_name
        if self.model_name == "resnet50":
            self.model, self.feature_dim, self.interm_feature_dim = resnet_50(
                pretrained=pretrained
            )
        elif self.model_name == "vit_base":
            vit_grad_ckpt = False
            vit_ckpt_layer = 0
            image_size = 224

            vit_name = self.model_name[4:]
            self.model, vision_width = create_vit(
                vit_name, image_size, vit_grad_ckpt, vit_ckpt_layer, 0)

            self.feature_dim = vision_width

            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            msg = self.model.load_state_dict(state_dict, strict=False)
            print(msg)
            self.feature_dim = vision_width
            self.interm_feature_dim = vision_width

        self.global_embedder = nn.Linear(self.feature_dim, self.output_dim)
        self.local_embedder = nn.Conv2d(
            self.interm_feature_dim,
            self.output_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        if freeze_cnn:
            print("Freezing CNN model")
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x, get_local=False):
        # --> fixed-size input: batch x 3 x 299 x 299
        if self.model_name == "resnet50":
            global_ft, local_ft = self.resnet_forward(x, extract_features=True)
        elif self.model_name == "vit_base":
            global_ft, local_ft = self.vit_forward(x, extract_features=True)
        if get_local:
            return global_ft, local_ft
        else:
            return global_ft

    def generate_embeddings(self, global_features, local_features):

        global_emb = self.global_embedder(global_features)
        local_emb = self.local_embedder(local_features)

        return global_emb, local_emb

    def vit_forward(self, x, extract_features=False):
        features = self.model(x)
        global_ft = features[:, 0]
        local_ft = features[:, 1:]
        local_ft = rearrange(local_ft, "b (h w) c -> b c h w", h=14)
        return global_ft, local_ft

    def resnet_forward(self, x, extract_features=False):

        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode="bilinear",
                        align_corners=True)(x)

        x = self.model.conv1(x)  # (batch_size, 64, 150, 150)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)  # (batch_size, 64, 75, 75)
        x = self.model.layer2(x)  # (batch_size, 128, 38, 38)
        x = self.model.layer3(x)  # (batch_size, 256, 19, 19)
        local_features = x
        x = self.model.layer4(x)  # (batch_size, 512, 10, 10)

        x = self.pool(x)
        x = x.view(x.size(0), -1)

        return x, local_features

    def densenet_forward(self, x, extract_features=False):
        pass

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)


class PretrainedImageClassifier(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        num_cls: int,
        feature_dim: int,
        freeze_encoder: bool = True,
    ):
        super(PretrainedImageClassifier, self).__init__()
        self.img_encoder = image_encoder
        self.classifier = nn.Linear(feature_dim, num_cls)
        if freeze_encoder:
            for param in self.img_encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.img_encoder(x)
        pred = self.classifier(x)
        return pred


class BertEncoder(nn.Module):
    def __init__(self,
                 bert_type="emilyalsentzer/Bio_ClinicalBERT",
                 last_n_layers=4,
                 aggregate_method="sum",
                 norm=False,
                 embedding_dim=768,
                 freeze_bert=False,
                 agg_tokens=True):

        super(BertEncoder, self).__init__()

        self.bert_type = bert_type
        self.last_n_layers = last_n_layers
        self.aggregate_method = aggregate_method
        self.norm = norm
        self.embedding_dim = embedding_dim
        self.freeze_bert = freeze_bert
        self.agg_tokens = agg_tokens

        self.model = AutoModel.from_pretrained(
            self.bert_type, output_hidden_states=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

        self.emb_global, self.emb_local = None, None

        if self.freeze_bert is True:
            print("Freezing BERT model")
            for param in self.model.parameters():
                param.requires_grad = False

    def aggregate_tokens(self, embeddings, caption_ids):

        batch_size, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)
        agg_embs_batch = []
        sentences = []

        # loop over batch
        for embs, caption_id in zip(embeddings, caption_ids):

            agg_embs = []
            token_bank = []
            words = []
            word_bank = []

            # loop over sentence
            for word_emb, word_id in zip(embs, caption_id):

                word = self.idxtoword[word_id.item()]

                if word == "[SEP]":
                    new_emb = torch.stack(token_bank)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))

                    agg_embs.append(word_emb)
                    words.append(word)
                    break

                if not word.startswith("##"):
                    if len(word_bank) == 0:
                        token_bank.append(word_emb)
                        word_bank.append(word)
                    else:
                        new_emb = torch.stack(token_bank)
                        new_emb = new_emb.sum(axis=0)
                        agg_embs.append(new_emb)
                        words.append("".join(word_bank))

                        token_bank = [word_emb]
                        word_bank = [word]
                else:
                    if word.startswith("##"):
                        token_bank.append(word_emb)
                        word_bank.append(word[2:])

            agg_embs = torch.stack(agg_embs)
            padding_size = num_words - len(agg_embs)
            paddings = torch.zeros(padding_size, num_layers, dim)
            paddings = paddings.to(agg_embs.device)
            words = words + ["[PAD]"] * padding_size

            agg_embs_batch.append(torch.cat([agg_embs, paddings]))
            sentences.append(words)

        agg_embs_batch = torch.stack(agg_embs_batch)
        agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
        return agg_embs_batch, sentences

    def forward(self, ids, attn_mask):

        outputs = self.model(ids, attn_mask)

        # aggregate intermetidate layers
        if self.last_n_layers > 1:
            all_embeddings = outputs[2]
            embeddings = torch.stack(
                all_embeddings[-self.last_n_layers:]
            )  # layers, batch, sent_len, embedding size

            embeddings = embeddings.permute(1, 0, 2, 3)

            if self.agg_tokens:
                embeddings, sents = self.aggregate_tokens(embeddings, ids)
            else:
                sents = [[self.idxtoword[w.item()] for w in sent]
                         for sent in ids]

            sent_embeddings = embeddings.mean(axis=2)

            if self.aggregate_method == "sum":
                word_embeddings = embeddings.sum(axis=1)
                sent_embeddings = sent_embeddings.sum(axis=1)
            elif self.aggregate_method == "mean":
                word_embeddings = embeddings.mean(axis=1)
                sent_embeddings = sent_embeddings.mean(axis=1)
            else:
                print(self.aggregate_method)
                raise Exception("Aggregation method not implemented")

        # use last layer
        else:
            word_embeddings, sent_embeddings = outputs[0], outputs[1]

        batch_dim, num_words, feat_dim = word_embeddings.shape
        word_embeddings = word_embeddings.view(batch_dim * num_words, feat_dim)
        if self.emb_local is not None:
            word_embeddings = self.emb_local(word_embeddings)
        word_embeddings = word_embeddings.view(
            batch_dim, num_words, self.embedding_dim)
        word_embeddings = word_embeddings.permute(0, 2, 1)

        if self.emb_global is not None:
            sent_embeddings = self.emb_global(sent_embeddings)

        if self.norm is True:
            word_embeddings = word_embeddings / torch.norm(
                word_embeddings, 2, dim=1, keepdim=True
            ).expand_as(word_embeddings)
            sent_embeddings = sent_embeddings / torch.norm(
                sent_embeddings, 2, dim=1, keepdim=True
            ).expand_as(sent_embeddings)

        return word_embeddings, sent_embeddings, sents
