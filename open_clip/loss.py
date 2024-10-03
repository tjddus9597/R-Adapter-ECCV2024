import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss

class OurLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.temp = 0.07

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_all_logits(self, image_features, text_features, logit_scale, ignore_scale=False, detach=False):
        if ignore_scale:
            logit_scale = 1.0

        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            logits_per_i2i = logit_scale * all_image_features @ all_image_features.T
            logits_per_t2t = logit_scale * all_text_features @ all_text_features.T
            logits_per_t2i = logit_scale * all_text_features @ all_image_features.T
            logits_per_i2t = logit_scale * all_image_features @ all_text_features.T
        else:
            logits_per_i2i = logit_scale * image_features @ image_features.T
            logits_per_t2t = logit_scale * text_features @ text_features.T
            logits_per_t2i = logit_scale * text_features @ image_features.T
            logits_per_i2t = logit_scale * image_features @ text_features.T

        if detach:
            logits_per_i2i = logits_per_i2i.detach()
            logits_per_t2t = logits_per_t2t.detach()
            logits_per_t2i = logits_per_t2i.detach()
            logits_per_i2t = logits_per_i2t.detach()
        
        return logits_per_i2i, logits_per_t2t, logits_per_t2i, logits_per_i2t

    # def compute_contrast(self, logits, mask, student_logit_scale, teacher_logit_scale):
    #     # for numerical stability
    #     logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    #     logits = logits - logits_max.detach()
    #     # compute log_prob
    #     exp_logits = torch.exp(logits)
    #     log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    #     mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

    #     # loss
    #     loss = - (teacher_logit_scale / student_logit_scale) * mean_log_prob_pos
    #     loss = loss.mean()

    #     return loss

    def compute_contrast(self, logits, mask, student_logit_scale=1, teacher_logit_scale=1):
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1)

        loss = - (teacher_logit_scale / student_logit_scale) * mean_log_prob_pos
        loss = loss.mean()

        # pull_losses = torch.relu(1 - logits).pow(2) * mask
        # push_losses = torch.relu(logits).pow(2) * (1-mask)
        # loss = (pull_losses.sum() + push_losses.sum()) / len(logits)

        return loss

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        t_logits_per_i2i, t_logits_per_t2t, t_logits_per_t2i, t_logits_per_i2t = self.get_all_logits(dist_image_features, dist_text_features, 10, ignore_scale=False, detach=True)
        s_logits_per_i2i, s_logits_per_t2t, s_logits_per_t2i, s_logits_per_i2t = self.get_all_logits(image_features, text_features, 10, ignore_scale=True)

        # mask = (t_logits_per_i2i + t_logits_per_t2t) / 2
        mask = (t_logits_per_i2i.softmax(1) ** 0.25) * (t_logits_per_t2t.softmax(1) ** 0.25) * (t_logits_per_t2i.softmax(1) ** 0.25) * (t_logits_per_i2t.softmax(1) ** 0.25)
        mask = mask / mask.sum(1, keepdim=True)

        # mask = torch.stack([t_logits_per_i2i,t_logits_per_t2t,t_logits_per_t2i,t_logits_per_i2t], dim=2).max(dim=2)[0]
        
        total_loss = (self.compute_contrast(s_logits_per_t2i, mask, logit_scale, dist_logit_scale) +
                    self.compute_contrast(s_logits_per_i2t, mask, logit_scale, dist_logit_scale) 
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        clip_loss = super().forward(image_features, text_features, logit_scale)
        clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss
