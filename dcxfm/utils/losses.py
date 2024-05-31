# # import tensorflow as tf
# # import torcheras.bactorchend as torch
# # from torcheras.losses import binary_crossentropy, BinaryCrossentropy
# import torch 
# import torch.nn as nn
# import torch.nn.functional as F

# beta = 0.25
# alpha = 0.25
# gamma = 2
# epsilon = 1e-5
# smooth = 1

# def dice_coef(y_true, y_pred):
#     y_true_f = torch.flatten(y_true)
#     y_pred_f = torch.flatten(y_pred)
#     intersection = torch.sum(y_true_f * y_pred_f)
#     return (2. * intersection + epsilon) / (
#             torch.sum(y_true_f) + torch.sum(y_pred_f) + epsilon)

# def sensitivity(y_true, y_pred):
#     true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
#     possible_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
#     return true_positives / (possible_positives + epsilon)

# def specificity(y_true, y_pred):
#     true_negatives = torch.sum(
#         torch.round(torch.clip((1 - y_true) * (1 - y_pred), 0, 1)))
#     possible_negatives = torch.sum(torch.round(torch.clip(1 - y_true, 0, 1)))
#     return true_negatives / (possible_negatives + epsilon)

# def convert_to_logits(y_pred):
#     y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
#     return torch.log(y_pred / (1 - y_pred))

# def weighted_cross_entropyloss(y_true, y_pred):
#     y_pred = convert_to_logits(y_pred)
#     pos_weight = beta / (1 - beta)
#     loss = F.cross_entropy(input=y_pred,
#                            targets=y_true,
#                            pos_weight=pos_weight)
#     return torch.mean(loss)

# def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
#     weight_a = alpha * (1 - y_pred) ** gamma * targets
#     weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

#     return (torch.log1p(torch.exp(-torch.abs(logits))) + F.relu(
#         -logits)) * (weight_a + weight_b) + logits * weight_b

# def focal_loss(y_true, y_pred):
#     y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
#     logits = torch.log(y_pred / (1 - y_pred))

#     loss = focal_loss_with_logits(logits=logits, targets=y_true,
#                                   alpha=alpha, gamma=gamma, y_pred=y_pred)

#     return torch.mean(loss)

# def depth_softmax(matrix):
#     sigmoid = lambda x: 1 / (1 + torch.exp(-x))
#     sigmoided_matrix = sigmoid(matrix)
#     softmax_matrix = sigmoided_matrix / torch.sum(sigmoided_matrix, axis=0)
#     return softmax_matrix

# def generalized_dice_coefficient(y_true, y_pred):
#     smooth = 1.
#     y_true_f = torch.flatten(y_true)
#     y_pred_f = torch.flatten(y_pred)
#     intersection = torch.sum(y_true_f * y_pred_f)
#     score = (2. * intersection + smooth) / (
#             torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
#     return score

# def dice_loss(y_true, y_pred):
#     loss = 1 - generalized_dice_coefficient(y_true, y_pred)
#     return loss

# def bce_dice_loss(y_true, y_pred):
#     loss = F.cross_entropy(y_true, y_pred) + dice_loss(y_true, y_pred)
#     return loss / 2.0

# def confusion(y_true, y_pred):
#     smooth = 1
#     y_pred_pos = torch.clip(y_pred, 0, 1)
#     y_pred_neg = 1 - y_pred_pos
#     y_pos = torch.clip(y_true, 0, 1)
#     y_neg = 1 - y_pos
#     tp = torch.sum(y_pos * y_pred_pos)
#     fp = torch.sum(y_neg * y_pred_pos)
#     fn = torch.sum(y_pos * y_pred_neg)
#     prec = (tp + smooth) / (tp + fp + smooth)
#     recall = (tp + smooth) / (tp + fn + smooth)
#     return prec, recall

# def true_positive(y_true, y_pred):
#     smooth = 1
#     y_pred_pos = torch.round(torch.clip(y_pred, 0, 1))
#     y_pos = torch.round(torch.clip(y_true, 0, 1))
#     tp = (torch.sum(y_pos * y_pred_pos) + smooth) / (torch.sum(y_pos) + smooth)
#     return tp

# def true_negative(y_true, y_pred):
#     smooth = 1
#     y_pred_pos = torch.round(torch.clip(y_pred, 0, 1))
#     y_pred_neg = 1 - y_pred_pos
#     y_pos = torch.round(torch.clip(y_true, 0, 1))
#     y_neg = 1 - y_pos
#     tn = (torch.sum(y_neg * y_pred_neg) + smooth) / (torch.sum(y_neg) + smooth)
#     return tn

# def tverstorchy_index(y_true, y_pred):
#     y_true_pos = torch.flatten(y_true)
#     y_pred_pos = torch.flatten(y_pred)
#     true_pos = torch.sum(y_true_pos * y_pred_pos)
#     false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
#     false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
#     alpha = 0.7
#     return (true_pos + smooth) / (true_pos + alpha * false_neg + (
#             1 - alpha) * false_pos + smooth)

# def tverstorchy_loss(y_true, y_pred):
#     return 1 - tverstorchy_index(y_true, y_pred)

# def focal_tverstorchy(y_true, y_pred):
#     pt_1 = tverstorchy_index(y_true, y_pred)
#     gamma = 0.75
#     return torch.pow((1 - pt_1), gamma)

# def log_cosh_dice_loss(y_true, y_pred):
#     x = dice_loss(y_true, y_pred)
#     return torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)

# def jacard_similarity(y_true, y_pred):
#     """
#         Intersection-Over-Union (IoU), also torchnown as the Jaccard Index
#     """
#     y_true_f = torch.flatten(y_true)
#     y_pred_f = torch.flatten(y_pred)

#     intersection = torch.sum(y_true_f * y_pred_f)
#     union = torch.sum((y_true_f + y_pred_f) - (y_true_f * y_pred_f))
#     return intersection / union

# def jacard_loss(y_true, y_pred):
#     """
#         Intersection-Over-Union (IoU), also torchnown as the Jaccard loss
#     """
#     return 1 - jacard_similarity(y_true, y_pred)


# def gaussian(window_size, sigma):
#     gauss = torch.Tensor([torch.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
#     return gauss/gauss.sum()


# def create_window(window_size, channel):
#     _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#     window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
#     return window


# def _ssim(img1, img2, window, window_size, channel, size_average = True):
#     mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
#     mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1*mu2

#     sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
#     sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

#     C1 = 0.01**2
#     C2 = 0.03**2

#     ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

#     if size_average:
#         return ssim_map.mean()
#     else:
#         return ssim_map.mean(1).mean(1).mean(1)

# class SSIM(torch.nn.Module):
#     def __init__(self, window_size = 11, size_average = True):
#         super(SSIM, self).__init__()
#         self.window_size = window_size
#         self.size_average = size_average
#         self.channel = 1
#         self.window = create_window(window_size, self.channel)

#     def forward(self, img1, img2):
#         (_, channel, _, _) = img1.size()

#         if channel == self.channel and self.window.data.type() == img1.data.type():
#             window = self.window
#         else:
#             window = create_window(self.window_size, channel)
            
#             if img1.is_cuda:
#                 window = window.cuda(img1.get_device())
#             window = window.type_as(img1)
            
#             self.window = window
#             self.channel = channel

#         return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

# def ssim(img1, img2, window_size = 11, size_average = True):
#     (_, channel, _, _) = img1.size()
#     window = create_window(window_size, channel)
    
#     if img1.is_cuda:
#         window = window.cuda(img1.get_device())
#     window = window.type_as(img1)
    
#     return _ssim(img1, img2, window, window_size, channel, size_average)

# def ssim_loss(y_true, y_pred):
#     """
#     Structural Similarity Index (SSIM) loss
#     """
#     return -ssim(y_true, y_pred)

# def unet3p_hybrid_loss(y_true, y_pred):
#     """
#     Hybrid loss proposed in UNET 3+ (https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf)
#     Hybrid loss for segmentation in three-level hierarchy – pixel, patch and map-level,
#     which is able to capture both large-scale and fine structures with clear boundaries.
#     """
#     cur_focal_loss = focal_loss(y_true, y_pred)
#     ms_ssim_loss = ssim_loss(y_true, y_pred)
#     cur_jacard_loss = jacard_loss(y_true, y_pred)

#     return cur_focal_loss + ms_ssim_loss + cur_jacard_loss

# def basnet_hybrid_loss(y_true, y_pred):
#     """
#     Hybrid loss proposed in BASNET (https://arxiv.org/pdf/2101.04704.pdf)
#     The hybrid loss is a combination of the binary cross entropy, structural similarity
#     and intersection-over-union losses, which guide the networtorch to learn
#     three-level (i.e., pixel-, patch- and map- level) hierarchy representations.
#     """
#     bce_loss = nn.CrossEntropyLoss()
#     cur_bce_loss = bce_loss(y_true, y_pred)

#     ms_ssim_loss = ssim_loss(y_true, y_pred)
#     cur_jacard_loss = jacard_loss(y_true, y_pred)

#     return cur_bce_loss + ms_ssim_loss + cur_jacard_loss