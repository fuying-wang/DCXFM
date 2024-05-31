import torch
import math
from torch import nn
from typing import List
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange
import ipdb
from transformers import AutoModel, AutoTokenizer
from dcxfm.modeling.sdmp.ibot_modules import iBOTHead
from dcxfm.modeling.sdmp.base_module import BaseLightningModule


class SDMPModule(BaseLightningModule):
    def __init__(self,
                 vision_model_name: str = "microsoft/swin-base-patch4-window12-384",
                 text_model_name: str = "microsoft/BiomedVLP-CXR-BERT-general",
                 dataset_dir: str = "/data1/r20user2/CXR_dataset",
                 dataset_list: List = ["mimic-cxr"],
                 logit_scale_init_value: float = 0.07,
                 train_data_pct: float = 1.,
                 loss_type: str = "nt_xent",
                 prompt_ensemble: bool = True,
                 imagesize: int = 512,
                 proj_dim: int = 128,
                 use_patch_cont: bool = False,
                 use_i2t_loss: bool = True,
                 use_self_distil_loss: bool = False,
                 use_local_loss: bool = False,
                 global_crops_number: int = 2,
                 local_crops_number: int = 4,
                 global_crops_size: int = 512,
                 local_crops_size: int = 224,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 encoder_momentum: float = 0.994,
                 center_momentum: float = 0.9,
                 teacher_temp: float = 0.07,
                 teacher_patch_temp: float = 0.07,
                 student_temp: float = 0.07,
                 patch_cont_temp: float = 0.1,
                 lambda1: float = 1.,
                 lambda2: float = 1.,
                 lambda3: float = 1.,
                 lr: float = 2e-5,
                 weight_decay: float = 1e-4,
                 max_epochs: int = 100,
                 num_devices: int = 1,
                 accumulate_grad_batches: int = 1,
                 *args,
                 **kwargs):

        super().__init__(
            vision_model_name=vision_model_name,
            text_model_name=text_model_name,
            dataset_dir=dataset_dir,
            dataset_list=dataset_list,
            train_data_pct=train_data_pct,
            prompt_ensemble=prompt_ensemble,
            imagesize=imagesize,
            global_crops_number=global_crops_number,
            local_crops_number=local_crops_number,
            global_crops_size=global_crops_size,
            local_crops_size=local_crops_size,
            batch_size=batch_size,
            num_workers=num_workers,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            num_devices=num_devices,
            accumulate_grad_batches=accumulate_grad_batches
        )

        self.save_hyperparameters()

        # self.logit_scale = nn.Parameter(
        #     torch.log(torch.tensor(1/logit_scale_init_value)))
        self.logit_scale_init_value = logit_scale_init_value
        self.loss_type = loss_type
        self.proj_dim = proj_dim
        self.use_patch_cont = use_patch_cont
        self.use_i2t_loss = use_i2t_loss
        self.use_self_distil_loss = use_self_distil_loss
        self.use_local_loss = use_local_loss
        # assert self.use_patch_cont or self.use_i2t_loss, "At least one loss should be used."
        self.encoder_momentum = encoder_momentum
        self.center_momentum = center_momentum
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.teacher_patch_temp = teacher_patch_temp
        self.patch_cont_temp = patch_cont_temp
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        # if not use self-distillation, we only use global crops
        if not self.use_self_distil_loss:
            self.local_crops_number = 0
        self.setup_datamodule()

        # Init important modules for iBoT
        self.img_encoder_teacher, num_fts, self.img_projection_teacher = self.init_img_encoder()
        self.img_encoder_student, _, self.img_projection_student = self.init_img_encoder()

        # TODO: Here we still keep iBOT head while it should be changed to DINO head
        self.ibot_head_teacher = iBOTHead(self.proj_dim,
                                          self.proj_dim,
                                          patch_out_dim=self.proj_dim,
                                          norm=None,
                                          act="gelu",
                                          shared_head=False,
                                          norm_last_layer=True)
        self.ibot_head_student = iBOTHead(self.proj_dim,
                                          self.proj_dim,
                                          patch_out_dim=self.proj_dim,
                                          norm=None,
                                          act="gelu",
                                          shared_head=False)

        self.tokenizer, self.text_encoder_student, self.text_projection_student = self.init_text_encoder()

        # Synchronize teacher and student
        self.img_encoder_teacher.load_state_dict(
            self.img_encoder_student.state_dict())
        self.img_projection_teacher.load_state_dict(
            self.img_projection_student.state_dict())
        self.ibot_head_teacher.load_state_dict(
            self.ibot_head_student.state_dict())
        
        # Freeze Teacher
        for param in self.img_encoder_teacher.parameters():
            param.requires_grad = False
        for param in self.img_projection_teacher.parameters():
            param.requires_grad = False
        for param in self.ibot_head_teacher.parameters():
            param.requires_grad = False

        self.register_buffer("center", torch.zeros(1, self.proj_dim))

    def init_img_encoder(self):
        # define vision model
        if self.vision_model_name == "resnet50":
            # self.imagesize = 512
            from dcxfm.modeling.sdmp.custom_resnet import resnet50
            vision_model = resnet50(pretrained=True)
            num_fts = vision_model.fc.in_features

        elif self.vision_model_name == "microsoft/swin-base-patch4-window12-384":
            self.imagesize = 384
            vision_model = AutoModel.from_pretrained(self.vision_model_name,
                                                     trust_remote_code=True)
            num_fts = 1024

        else:
            raise NotImplementedError

        img_projection_layer = nn.Linear(num_fts, self.proj_dim)

        return vision_model, num_fts, img_projection_layer

    def init_text_encoder(self):
        if self.text_model_name == "microsoft/BiomedVLP-CXR-BERT-general":
            text_model = AutoModel.from_pretrained(self.text_model_name,
                                                   trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(
                self.text_model_name, trust_remote_code=True)
            text_projection_layer = nn.Linear(768, self.proj_dim)
        else:
            raise NotImplementedError

        # return tokenizer, text_model, text_projection_layer, ibot_text_head
        return tokenizer, text_model, text_projection_layer

    def encode_text_student(self, input_ids=None, attention_mask=None):
        if self.text_model_name == "microsoft/BiomedVLP-CXR-BERT-general":
            text_outputs = self.text_encoder_student(
                input_ids=input_ids,
                attention_mask=attention_mask)
            word_feats = text_outputs.last_hidden_state

            cap_lens = []
            all_word_embeds = []
            all_text_embeds = []
            for i in range(len(input_ids)):
                cur_report = self.tokenizer.decode(input_ids[i])
                cur_words = cur_report.split()
                cur_cap_len = len(
                    [w for w in cur_words if not w.startswith("[")])
                cap_lens.append(cur_cap_len)
                cur_word_embeds = self.text_projection_student(word_feats[i])
                cur_text_embeds = torch.mean(
                    cur_word_embeds[1:cur_cap_len+1], dim=0)
                all_word_embeds.append(cur_word_embeds)
                all_text_embeds.append(cur_text_embeds)
            all_word_embeds = torch.stack(all_word_embeds, dim=0)
            all_text_embeds = torch.stack(all_text_embeds, dim=0)

        else:
            raise NotImplementedError

        return cap_lens, all_text_embeds, all_word_embeds

    def encode_image_teacher(self, pixel_values=None, dense=False):
        # image encoder
        if self.vision_model_name == "resnet50":
            vision_output = self.img_encoder_teacher(pixel_values)
            if dense:
                patch_feats = rearrange(vision_output, 'b c h w -> b (h w) c')
            img_feats = F.adaptive_avg_pool2d(vision_output, (1, 1))
            img_feats = img_feats.view(img_feats.size(0), -1)

        elif self.vision_model_name == "microsoft/swin-base-patch4-window12-384":
            output = self.img_encoder_teacher(pixel_values)
            if dense:
                patch_feats = output.last_hidden_state
            else:
                img_feats = output['pooler_output']

        if dense:
            return img_feats, patch_feats
        else:
            return img_feats

    def encode_image_student(self, pixel_values=None, dense=False):
        # image encoder
        if self.vision_model_name == "resnet50":
            vision_output = self.img_encoder_student(pixel_values)
            if dense:
                patch_feats = rearrange(vision_output, 'b c h w -> b (h w) c')
            img_feats = F.adaptive_avg_pool2d(vision_output, (1, 1))
            img_feats = img_feats.view(img_feats.size(0), -1)

        elif self.vision_model_name == "microsoft/swin-base-patch4-window12-384":
            output = self.img_encoder_student(pixel_values)
            if dense:
                patch_feats = output.last_hidden_state
            else:
                img_feats = output['pooler_output']

        if dense:
            return img_feats, patch_feats
        else:
            return img_feats

    @torch.no_grad()
    def _momentum_update_teacher_network(self) -> None:
        """Momentum update of the key encoder."""
        momentum = self.encoder_momentum
        for param_q, param_k in zip(self.img_encoder_student.parameters(), self.img_encoder_teacher.parameters()):
            param_k.data = param_k.data * momentum + \
                param_q.data * (1.0 - momentum)

        for param_q, param_k in zip(self.img_projection_student.parameters(), self.img_projection_teacher.parameters()):
            param_k.data = param_k.data * momentum + \
                param_q.data * (1.0 - momentum)

        for param_q, param_k in zip(self.ibot_head_student.parameters(), self.ibot_head_teacher.parameters()):
            param_k.data = param_k.data * momentum + \
                param_q.data * (1.0 - momentum)

    @torch.no_grad()
    # def update_center(self, teacher_cls, teacher_patch):
    def update_center(self, teacher_cls):
        """
        Update center used for teacher output.
        """
        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        torch.distributed.all_reduce(cls_center)
        cls_center = cls_center / \
            (len(teacher_cls) * torch.distributed.get_world_size())
        self.center = self.center * self.center_momentum + \
            cls_center * (1 - self.center_momentum)

    def encode_report(self, input_ids: List, attention_mask: List, is_report_list: bool = True):
        ''' Encode the radiology report
        Note that each report may consist of different number of sentences across reports
        Current strategy is to (weighted) average the sentence embeddings to obtain global embedding
        '''
        batch_size = len(input_ids)
        input_ids = rearrange(torch.stack(input_ids), "b n d -> (b n) d")
        attention_mask = rearrange(torch.stack(
            attention_mask), "b n d -> (b n) d")

        cap_lens, text_embeds_student, word_embeds_student = self.encode_text_student(
            input_ids, attention_mask)  # (B, D)
        # text_feats_student = rearrange(
        #     text_feats_student, "(b n) d -> b n d", b=batch_size)
        text_embeds_student = rearrange(
            text_embeds_student, "(b n) d -> b n d", b=batch_size)
        word_embeds_student = rearrange(
            word_embeds_student, "(b n) l d -> b n l d", b=batch_size)

        return cap_lens, text_embeds_student, word_embeds_student

    def forward(self,
                input_ids=None,
                img_global=None,
                img_local=None,
                attention_mask=None,
                return_loss=False,
                img_labels=None,
                text_labels=None,
                **kwargs,
                ):
        
        batch_size = img_global.size(0)
        # Forward text
        if return_loss:
            cap_lens, text_embeds_student, word_embeds_student = self.encode_report(
                input_ids, attention_mask)

            batch_img_global = rearrange(
                img_global, 'b n c h w -> (b n) c h w')
            global_crops_num = img_global.size(1)
            if img_local is not None:
                batch_img_local = rearrange(
                    img_local, 'b n c h w -> (b n) c h w')
                local_crops_num = img_local.size(1)
            else:
                local_crops_num = 0

            if self.use_self_distil_loss:
                # Forward global crops into teacher network
                img_feats_global_teacher, patch_feats_global_teacher = self.encode_image_teacher(
                    batch_img_global, dense=True)

                patch_embeds_global_teacher = self.img_projection_teacher(
                    patch_feats_global_teacher)
                img_embeds_global_teacher = torch.mean(
                    patch_embeds_global_teacher, dim=1)

                # For DINO head
                img_distr_global_teacher = self.ibot_head_teacher(
                    img_embeds_global_teacher)

                # Forward global crops into student network
                img_feats_global_student, patch_feats_global_student = self.encode_image_student(
                    batch_img_global, dense=True)
                patch_embeds_global_student = self.img_projection_student(
                    patch_feats_global_student)
                img_embeds_global_student = torch.mean(
                    patch_embeds_global_student, dim=1)
                # For DINO head
                img_distr_global_student = self.ibot_head_student(
                    img_embeds_global_student)
                
                # Get embeddings of local crop
                # Student use both the global and local crops
                img_feats_local_student, patch_feats_local_student = self.encode_image_student(
                    batch_img_local, dense=True)
                patch_embeds_local_student = self.img_projection_student(
                    patch_feats_local_student)
                img_embeds_local_student = torch.mean(
                    patch_embeds_local_student, dim=1)
                # For DINO head
                img_distr_local_student = self.ibot_head_student(
                    img_embeds_local_student)

                student_emb_g = rearrange(
                    img_embeds_global_student, "(b n) d -> b n d", b=batch_size).split(1, dim=1)
                student_emb_g = [x.squeeze(1) for x in student_emb_g]

                # Compute probability distribution of teacher output
                teacher_cls_c = F.softmax(
                    (img_distr_global_teacher - self.center) / self.teacher_temp, dim=-1)
                teacher_cls_c = rearrange(
                    teacher_cls_c.detach(), "(b n) d -> b n d", b=batch_size).split(1, dim=1)
                teacher_cls_c = [x.squeeze(1) for x in teacher_cls_c]

                # Compute probability distribution of student output
                student_cls_g = img_distr_global_student / self.student_temp
                student_cls_g = rearrange(
                    student_cls_g, "(b n) d -> b n d", b=batch_size).split(1, dim=1)
                student_cls_g = [x.squeeze(1) for x in student_cls_g]

                student_cls_l = img_distr_local_student / self.student_temp
                student_cls_l = rearrange(
                    student_cls_l, "(b n) d -> b n d", b=batch_size).split(1, dim=1)
                student_cls_l = [x.squeeze(1) for x in student_cls_l]

                student_cls_c = student_cls_g + student_cls_l
                student_patch_emb_g = rearrange(
                    patch_embeds_global_student, "(b n) l d -> b n l d", b=batch_size).split(1, dim=1)
                student_patch_emb_g = [x.squeeze(1)
                                       for x in student_patch_emb_g]

            else:
                # Forward global crops into student network
                img_feats_global_student, patch_feats_global_student = self.encode_image_student(
                    batch_img_global, dense=True)
                patch_embeds_global_student = self.img_projection_student(
                    patch_feats_global_student)
                img_embeds_global_student = torch.mean(
                    patch_embeds_global_student, dim=1)

                student_emb_g = rearrange(
                    img_embeds_global_student, "(b n) d -> b n d", b=batch_size).split(1, dim=1)
                student_emb_g = [x.squeeze(1) for x in student_emb_g]

                student_patch_emb_g = rearrange(
                    patch_embeds_global_student, "(b n) l d -> b n l d", b=batch_size).split(1, dim=1)
                student_patch_emb_g = [x.squeeze(1)
                                       for x in student_patch_emb_g]

            # distill loss
            total_loss1, n_loss_terms1 = 0, 0
            # cont loss
            total_loss2, n_loss_terms2 = 0, 0
            # local loss
            total_loss3, n_loss_terms3 = 0, 0
            # E.g., 2
            for q in range(global_crops_num):
                # E.g., 6
                for v in range(global_crops_num + local_crops_num):
                    if v == q:
                        # In this case, only consider the CL
                        if self.use_i2t_loss:
                            # logits is computed in the teacher network
                            logits_per_image_0 = self.compute_logits(
                                student_emb_g[v], text_embeds_student[:, 0])
                            # logits_per_image_1 = self.compute_logits(
                            #     student_emb_g[v], text_embeds_student[:, 1])
                            if self.loss_type == "nt_xent":
                                # loss2 = self._clip_loss(logits_per_image_0) + self._clip_loss(logits_per_image_1)
                                loss2 = self._clip_loss(logits_per_image_0)
                            elif self.loss_type == "soft_cont":
                                label_sim = torch.matmul(
                                    img_labels, text_labels.T)
                                # loss2 = self._soft_clip_loss(logits_per_image_0, label_sim) + self._soft_clip_loss(
                                #     logits_per_image_1, label_sim)
                                loss2 = self._soft_clip_loss(
                                    logits_per_image_0, label_sim)
                            else:
                                raise NotImplementedError(
                                    f"No such loss type {self.loss_type}")

                            total_loss2 += loss2
                            n_loss_terms2 += 1
                        else:
                            total_loss2 += torch.tensor(
                                0.).type_as(text_embeds_student)
                            n_loss_terms2 += 1

                        if self.use_local_loss:
                            # cap_lens = []
                            # for i in range(len(input_ids)):
                            #     cur_report = self.tokenizer.decode(input_ids[i][0])
                            #     cur_words = cur_report.split()
                            #     cap_lens.append(len([w for w in cur_words if not w.startswith("[")]))
                            h = w = int(
                                math.sqrt(student_patch_emb_g[q].shape[1]))
                            patch_embeds = rearrange(
                                student_patch_emb_g[q], "b (h w) d -> b d h w", h=h, w=w)
                            # only consider token embeddings
                            word_embeds = rearrange(
                                word_embeds_student[:, 0, 1:], "b n d -> b d n")
                            loss0, loss1, _ = self.local_loss(
                                patch_embeds, word_embeds, cap_lens=cap_lens)
                            total_loss3 += (loss0 + loss1) / 2
                            n_loss_terms3 += 1
                        else:
                            total_loss3 += torch.tensor(
                                0.).type_as(text_embeds_student)
                            n_loss_terms3 += 1
                    else:
                        # FIXME: check this part.
                        if self.use_self_distil_loss:
                            # Here we only consider CLS-level distillation
                            loss1 = torch.sum(-teacher_cls_c[q] * F.log_softmax(
                                student_cls_c[v], dim=-1), dim=-1)
                            total_loss1 += loss1.mean()
                            n_loss_terms1 += 1
                        else:
                            total_loss1 += torch.tensor(
                                0.).type_as(text_embeds_student)
                            n_loss_terms1 += 1

            if n_loss_terms1 > 0:
                total_loss1 = total_loss1 / n_loss_terms1 * self.lambda1
            if n_loss_terms2 > 0:
                total_loss2 = total_loss2 / n_loss_terms2 * self.lambda2
            if n_loss_terms3 > 0:
                total_loss3 = total_loss3 / n_loss_terms3 * self.lambda3
            # if n_loss_terms4 > 0:
            #     total_loss4 = total_loss4 / n_loss_terms4 * self.lambda4

            # total_loss = total_loss1 + total_loss2 + total_loss3 + total_loss4
            total_loss = total_loss1 + total_loss2 + total_loss3

            # It only happens in the traning stage
            if self.use_self_distil_loss and self.training:
                # text_distri_teacher = rearrange(
                #     text_distri_teacher, 'b n d -> (b n) d')
                self.update_center(img_distr_global_teacher)

            return {
                "loss": total_loss,
                "local_loss": total_loss3,
                "cont_loss": total_loss2,
                "distill_loss": total_loss1,
            }

        else:
            # FIXME: need to fix this part later
            # Because we use text embeddings of the teacher network in the test stage.
            # Embeddings in the teacher network are used for evaluation.
            text_embeds_teacher, _ = self.encode_report(
                input_ids, attention_mask)
            img_feats, patch_feats = self.encode_image_teacher(
                img_global, dense=True)  # (B, D), (B, N, D)
            img_embeds = self.img_projection_teacher(img_feats)
            patch_embeds = self.patch_projection_teacher(patch_feats)
            if self.use_patch_cont:
                img_embeds = self.compute_patch_cont_embeds(
                    patch_embeds, text_embeds_teacher)
            logits_per_image = self.compute_logits(
                img_embeds, text_embeds_teacher)
            logits_per_text = logits_per_image.t()

            return {'patch_embeds': patch_embeds, 'text_embeds': text_embeds_teacher,
                    'logits': logits_per_image,
                    'logits_per_text': logits_per_text}

    def compute_logits(self, img_emb, text_emb):

        img_emb = F.normalize(img_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)
        # we are not using learnable
        logits_per_text = torch.matmul(
            text_emb, img_emb.t()) / self.logit_scale_init_value

        return logits_per_text.t()

    def _clip_loss(self, logits):
        # Note that temperature has been divided

        labels = torch.arange(logits.size(0)).type_as(logits).long()
        loss0 = F.cross_entropy(logits, labels)
        loss1 = F.cross_entropy(logits.t(), labels)
        loss = (loss0 + loss1) / 2

        return loss

    def _soft_clip_loss(self, logits_per_img, soft_label):
        '''take labels of images and sentences as a softlabel
        e.g., image_label = [1, 0, 1, -1], sentence_label = [0, 0, 1, -1]
        this pair has similarity as: 1 * 0 + 0 * 0 + 1 * 1 + -1 * -1 = 2.
        We will clamp the similarity into [-1,1], and take softmax as a soft-label.
        '''
        # when using InfoNCE-like loss
        image_loss = self._soft_xent_loss(
            logits_per_img, F.softmax(soft_label / 0.5, 1))
        caption_loss = self._soft_xent_loss(
            logits_per_img.T, F.softmax(soft_label.T / 0.5, 1))

        return (image_loss + caption_loss) / 2.

    def _soft_xent_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax(input, dim=1)
        return -(target * logprobs).sum() / input.shape[0]

    @staticmethod
    def cosine_similarity(x1, x2, dim=1, eps=1e-8):
        """Returns cosine similarity between x1 and x2, computed along dim."""
        w12 = torch.sum(x1 * x2, dim)
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

    @staticmethod
    def attention_fn(query, context, temp1):
        """
        query: batch x ndf x queryL
        context: batch x ndf x ih x iw (sourceL=ihxiw)
        mask: batch_size x sourceL
        """
        batch_size, queryL = query.size(0), query.size(2)
        ih, iw = context.size(2), context.size(3)
        sourceL = ih * iw

        # --> batch x sourceL x ndf
        context = context.view(batch_size, -1, sourceL)
        contextT = torch.transpose(context, 1, 2).contiguous()

        # Get attention
        # (batch x sourceL x ndf)(batch x ndf x queryL)
        # -->batch x sourceL x queryL
        attn = torch.bmm(contextT, query)
        # --> batch*sourceL x queryL
        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.Softmax(dim=-1)(attn)

        # --> batch x sourceL x queryL
        attn = attn.view(batch_size, sourceL, queryL)
        # --> batch*queryL x sourceL
        attn = torch.transpose(attn, 1, 2).contiguous()
        attn = attn.view(batch_size * queryL, sourceL)

        attn = attn * temp1
        attn = nn.Softmax(dim=-1)(attn)
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attnT = torch.transpose(attn, 1, 2).contiguous()

        # (batch x ndf x sourceL)(batch x sourceL x queryL)
        # --> batch x ndf x queryL
        weightedContext = torch.bmm(context, attnT)

        return weightedContext, attn.view(batch_size, -1, ih, iw)

    def local_loss(
        self, img_features, words_emb, cap_lens, temp1=4.0, temp2=5.0, temp3=10.0, agg="sum"
    ):
        batch_size = img_features.shape[0]
        # TODO: Add these two normalization, it doesn't work ...
        # img_features = F.normalize(img_features, p=2, dim=1)
        # words_emb = F.normalize(words_emb, p=2, dim=1)

        att_maps = []
        similarities = []
        # cap_lens = cap_lens.data.tolist()
        for i in range(words_emb.shape[0]):

            # Get the i-th text description
            words_num = cap_lens[i]  # 25
            # TODO: remove [SEP]
            # word = words_emb[i, :, 1:words_num+1].unsqueeze(0).contiguous()    # [1, 768, 25]
            word = words_emb[i, :, :words_num].unsqueeze(
                0).contiguous()  # [1, 768, 25]
            word = word.repeat(batch_size, 1, 1)  # [48, 768, 25]
            context = img_features  # [48, 768, 19, 19]

            weiContext, attn = self.attention_fn(
                word, context, temp1
            )  # [48, 768, 25], [48, 25, 19, 19]

            att_maps.append(
                attn[i].unsqueeze(0).contiguous()
            )  # add attention for curr index  [25, 19, 19]
            word = word.transpose(1, 2).contiguous()  # [48, 25, 768]
            weiContext = weiContext.transpose(
                1, 2).contiguous()  # [48, 25, 768]

            word = word.view(batch_size * words_num, -1)  # [1200, 768]
            weiContext = weiContext.view(
                batch_size * words_num, -1)  # [1200, 768]

            row_sim = self.cosine_similarity(word, weiContext)
            row_sim = row_sim.view(batch_size, words_num)  # [48, 25]

            row_sim.mul_(temp2).exp_()
            if agg == "sum":
                row_sim = row_sim.sum(dim=1, keepdim=True)  # [48, 1]
            else:
                row_sim = row_sim.mean(dim=1, keepdim=True)  # [48, 1]
            row_sim = torch.log(row_sim)

            similarities.append(row_sim)

        similarities = torch.cat(similarities, 1)  #
        similarities = similarities * temp3
        similarities1 = similarities.transpose(0, 1)  # [48, 48]

        labels = Variable(torch.LongTensor(range(batch_size))
                          ).to(similarities.device)

        loss0 = nn.CrossEntropyLoss()(similarities, labels)  # labels: arange(batch_size)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)

        return loss0, loss1, att_maps
