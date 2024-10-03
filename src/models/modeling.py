import torch
import copy

import clip.clip as clip

from src.models import utils
import open_clip
from torch.nn import functional as F

def PEFT_config(args):
    def peft_init(args, idx):
        peft_config = {}
        peft_config['lora'] = args.lora[idx]
        peft_config['adapter'] = args.adapter[idx]
        peft_config['use_peft'] = args.use_peft
        peft_config['drop_path'] = args.drop_path
        peft_config['ema'] = args.ema
        peft_config['bma'] = args.bma
        peft_config['eval_scale'] = args.eval_scale
        return peft_config

    args.use_peft = any([
                        args.lora[0] > 0,
                        args.lora[1] > 0,
                        args.adapter[0] > 0,
                        args.adapter[1] > 0,
                        ])
    v_peft_config = peft_init(args, 0)
    t_peft_config = peft_init(args, 1)
    return v_peft_config, t_peft_config

class CLIPEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()
        peft_config = PEFT_config(args)
        self.model, self.train_preprocess, self.val_preprocess = clip.load(
            args.model, args.device, jit=False, precision=args.precision, 
            peft_config=peft_config)
        self.cache_dir = args.cache_dir

        self.model.lora_vision_unlock() if args.lora[0] > 0 else None
        self.model.lora_text_unlock() if args.lora[1] > 0 else None
        self.model.adapter_vision_unlock() if args.adapter[0] > 0 else None
        self.model.adapter_text_unlock() if args.adapter[1] > 0 else None
        self.model.lock_vision_tower() if args.lock_image else None
        self.model.lock_text_tower() if args.lock_text else None
        self.model.unlock_proj_image_text() if args.unlock_proj else None
        self.model.unlock_ln() if args.unlock_ln else None
        self.model.unlock_bias() if args.unlock_bias else None
        self.model.unlock_cls() if args.unlock_cls else None

    def forward(self, images, text=None, steps=None):
        assert self.model is not None
        return self.model(images, text, steps=steps)

    def save(self, filename):
        print(f'Saving clip encoder to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename, logger=None):
        print(f'Loading image encoder from {filename}')
        if logger != None:
            logger.info(f'Loading image encoder from {filename}')
        return utils.torch_load(filename)


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None, shape=[512, 1000]):
        if weights is not None:
            output_size, input_size = weights.shape
            super().__init__(input_size, output_size)
        else:
            super().__init__(shape[0], shape[1])
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        self.bias = None

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
            self.weight.data = F.normalize(self.weight.data, dim=-1)
        return super().forward(inputs)

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename, logger=None):
        print(f'Loading classification head from {filename}')
        if logger != None:
            logger.info(f'Loading classification head from {filename}')
        return utils.torch_load(filename)


class ImageClassifier(torch.nn.Module):
    def __init__(self,
                 image_encoder,
                 classification_head,
                 process_images=True):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        self.process_images = process_images
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def forward(self, inputs):
        if self.process_images:
            inputs = self.image_encoder(inputs)
        outputs = self.classification_head(inputs)
        return outputs

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)


class ImageClassifier_Norm(torch.nn.Module):
    def __init__(self,
                 image_encoder,
                 classification_head,
                 process_images=True):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        self.process_images = process_images
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def forward(self, inputs):
        if self.process_images:
            inputs = self.image_encoder(inputs)
        inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        outputs = self.classification_head(inputs)
        return outputs

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)


class ImageEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        peft_config = PEFT_config(args)
        self.model, self.train_preprocess, self.val_preprocess = clip.load(
            args.model, args.device, jit=False, precision=args.precision, 
            peft_config=peft_config)
        self.cache_dir = args.cache_dir

        self.model.lora_vision_unlock() if args.lora[0] > 0 else None
        self.model.lora_text_unlock() if args.lora[1] > 0 else None
        self.model.adapter_vision_unlock() if args.adapter[0] > 0 else None
        self.model.adapter_text_unlock() if args.adapter[1] > 0 else None
        self.model.lock_vision_tower() if args.lock_image else None
        self.model.lock_text_tower() if args.lock_text else None
        self.model.unlock_proj_image_text() if args.unlock_proj else None
        self.model.unlock_ln() if args.unlock_ln else None
        self.model.unlock_bias() if args.unlock_bias else None
        self.model.unlock_cls() if args.unlock_cls else None


        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image encoder from {filename}')
        return utils.torch_load(filename)