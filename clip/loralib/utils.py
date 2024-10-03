import torch
import torch.nn as nn
from typing import Dict
from .layers import LoRALayer
from .adapter_layers import *
from itertools import chain, repeat
import random

def mark_only_lora_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if 'lora_' in n and 'cache' not in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

def mark_only_adapter_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if 'adapter' in n and 'cache' not in n:
            p.requires_grad = True
        else:
            p.requires_grad = False


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError

def forward_vit_block_adapter(self, x):
    if not self.training and (self.ema or self.bma):
        adapter_attn, adapter_mlp = self.adapter_attn_cache, self.adapter_mlp_cache
    else:
        adapter_attn, adapter_mlp = self.adapter_attn, self.adapter_mlp

    if self.merged:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
    else:
        x = x + adapter_attn(self.attention(self.ln_1(x)))
        x = x + adapter_mlp(self.mlp(self.ln_2(x)))

    return x




def set_Adapter(blocks, dim, peft_config):
    for i, _ in enumerate(blocks.children()):
        if peft_config['adapter'] > 0:
            _.adapter_attn = StochasticAdapter(embed_dim=dim, r=peft_config['adapter'], drop_path=peft_config['drop_path'], eval_scale=peft_config['eval_scale'])
            _.adapter_mlp = StochasticAdapter(embed_dim=dim, r=peft_config['adapter'], drop_path=peft_config['drop_path'], eval_scale=peft_config['eval_scale'])
        else:
            _.adapter_attn, _.adapter_mlp = None, None

        with torch.no_grad():
            _.org_proj_weight, _.org_proj_bias = _.attn.out_proj.weight.data, _.attn.out_proj.bias.data
            _.org_fc2_weight, _.org_fc2_bias = _.mlp.c_proj.weight.data, _.mlp.c_proj.bias.data

        _.ema, _.bma = peft_config['ema'], peft_config['bma']
        _.merged = False

        if _.ema or _.bma:
            _.adapter_attn_cache = copy.deepcopy(_.adapter_attn)
            _.adapter_mlp_cache = copy.deepcopy(_.adapter_mlp)

        bound_method = forward_vit_block_adapter.__get__(_, _.__class__)
        setattr(_, 'forward', bound_method)


def reparameterize(Wa, Wb=None, Ba=None, Bb=None, scale=1, do_residual=False):
    bias = 0
    id_tensor=0
    if Ba is not None:
        bias=Ba@Wb
    if Bb is not None:
        bias=bias+Bb
    if do_residual:
        if Wb is not None:
            id_tensor=torch.eye(Wa.shape[0],Wb.shape[1]).to(Wa.device)
        else:
            id_tensor=torch.eye(Wa.shape[0],Wa.shape[1]).to(Wa.device)
    if Wb is not None:
        weight = Wa @ Wb * scale + id_tensor
    else:
        weight = Wa * scale + id_tensor
    return weight.T, bias*scale if isinstance(bias,torch.Tensor) else None

def Rep_AdaptWeight(model, args, eval_scale=0.5):
    encoders = []
    if args.adapter[0] > 0:
        encoders.append(model.visual.transformer)
    if args.adapter[1] > 0:
        encoders.append(model.transformer)

    for encoder in encoders:
        for i, _ in enumerate(encoder.resblocks.children()):
            _.merged = True

            with torch.no_grad():
                _.org_attn_weight, _.org_attn_bias = _.attn.out_proj.weight.data.clone().detach(), _.attn.out_proj.bias.data.clone().detach()
                _.org_mlp_weight, _.org_mlp_bias = _.mlp.c_proj.weight.data.clone().detach(), _.mlp.c_proj.bias.data.clone().detach()

            if _.ema or _.bma:
                adapter_attn, adapter_mlp = _.adapter_attn_cache, _.adapter_mlp_cache
            else:
                adapter_attn, adapter_mlp = _.adapter_attn, _.adapter_mlp

            adapt_attn_scale = adapter_attn.s * eval_scale

            if adapter_attn.embed_dim > adapter_attn.r:
                merged_weight, _ = reparameterize(adapter_attn.d.weight.squeeze().T, adapter_attn.u.weight.squeeze().T)
                adapt_attn_weight, adapt_attn_bias = reparameterize(merged_weight.squeeze().T, scale=adapt_attn_scale, do_residual=True)
            else:
                adapt_attn_weight, adapt_attn_bias = reparameterize(adapter_attn.f.weight.squeeze().T, scale=adapt_attn_scale, do_residual=True)

            rep_attn_weight, rep_attn_bias = reparameterize(_.attn.out_proj.weight.T, adapt_attn_weight.T, _.attn.out_proj.bias, adapt_attn_bias)

            adapt_mlp_scale = adapter_mlp.s * eval_scale
            
            if adapter_mlp.embed_dim > adapter_mlp.r:
                merged_weight, _ = reparameterize(adapter_mlp.d.weight.squeeze().T, adapter_mlp.u.weight.squeeze().T)
                adapt_mlp_weight, adapt_mlp_bias = reparameterize(merged_weight.squeeze().T, scale=adapt_mlp_scale, do_residual=True)
            else:
                adapt_mlp_weight, adapt_mlp_bias = reparameterize(adapter_mlp.f.weight.squeeze().T, scale=adapt_mlp_scale, do_residual=True)

            rep_mlp_weight, rep_mlp_bias = reparameterize( _.mlp.c_proj.weight.T, adapt_mlp_weight.T, _.mlp.c_proj.bias, adapt_mlp_bias)

            with torch.no_grad():
                _.attn.out_proj.weight.copy_(rep_attn_weight)
                _.attn.out_proj.bias.copy_(rep_attn_bias)
                _.mlp.c_proj.weight.copy_(rep_mlp_weight)
                _.mlp.c_proj.bias.copy_(rep_mlp_bias)

def Repback_AdaptWeight(model, args):
    encoders = []
    if args.adapter[0] > 0:
        encoders.append(model.visual.transformer)
    if args.adapter[1] > 0:
        encoders.append(model.transformer)

    for encoder in encoders:
        for i, _ in enumerate(encoder.resblocks.children()):
            _.merged = False
            with torch.no_grad():
                _.attn.out_proj.weight.copy_(_.org_attn_weight)
                _.attn.out_proj.bias.copy_(_.org_attn_bias)
                _.mlp.c_proj.weight.copy_(_.org_mlp_weight)
                _.mlp.c_proj.bias.copy_(_.org_mlp_bias)

def ema_update(args, model, m):
    encoders = []
    if args.adapter[0] > 0:
        encoders.append(model.visual.transformer)
    if args.adapter[1] > 0:
        encoders.append(model.transformer)
    for encoder in encoders:
        for i, _ in enumerate(encoder.resblocks.children()):
            avg_model_params = list(_.adapter_attn_cache.parameters()) + list(_.adapter_mlp_cache.parameters())
            model_params = list(_.adapter_attn.parameters()) + list(_.adapter_mlp.parameters())
            for moving_avg_param, param in zip(avg_model_params, model_params):
                moving_avg_param.data = m * moving_avg_param.data + (1-m) * param.data.detach()       

def bma_update(args, model, steps):
    it, total_iter = steps
    beta = self.v_peft_config['bma']
    beta_dist = sp.stats.beta(beta, beta)
    weight = beta_dist.pdf((it + 0.5) / (total_iter + 1))
    self.weight_sum += weight
    relative_weight = weight / self.weight_sum
    with torch.no_grad():
        for i in range(12):
            if self.v_peft_config['adapter'] > 0:
                avg_model_params = list(self.visual.transformer.resblocks[i].adapter_attn_cache.parameters()) + list(self.visual.transformer.resblocks[i].adapter_mlp_cache.parameters())
                model_params = list(self.visual.transformer.resblocks[i].adapter_attn.parameters()) + list(self.visual.transformer.resblocks[i].adapter_mlp.parameters())
                for moving_avg_param, param in zip(avg_model_params, model_params):
                    moving_avg_param.data = (moving_avg_param.data + relative_weight * param.data.detach()) / (1 + relative_weight)
            if self.t_peft_config['adapter'] > 0:
                avg_model_params = list(self.transformer.resblocks[i].adapter_attn_cache.parameters()) + list(self.transformer.resblocks[i].adapter_mlp_cache.parameters())
                model_params = list(self.transformer.resblocks[i].adapter_attn.parameters()) + list(self.transformer.resblocks[i].adapter_mlp.parameters())
                for moving_avg_param, param in zip(avg_model_params, model_params):
                    moving_avg_param.data = (moving_avg_param.data + relative_weight * param.data.detach()) / (1 + relative_weight)