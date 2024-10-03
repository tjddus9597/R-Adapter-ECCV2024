
# Running Experiments with R-Adapter

This document contains instructions on how to run experiments using the **R-Adapter** method.

## Experiments

### Full Fine-Tuning on ImageNet

#### Preparing the ImageNet Dataset

Link your ImageNet dataset to the expected directory:

```bash
ln -s /path/to/your/ILSVRC2012/dataset ./datasets/data/ILSVRC2012
```

Generate the CSV file required for data loading:

```bash
python datacreation_scripts/imagenet_csv_creator.py
```

#### Running the Fine-Tuning Script

```bash
torchrun --master_port 126 --nproc_per_node 4 -m src.main     --train-dataset=ImageNet     --epochs=10     --lr=5e-4     --wd=0     --workers=8     --batch-size=512     --model=ViT-B/16     --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetSketch,ObjectNet,ImageNetA     --template=openai_imagenet_template     --save=./checkpoints/     --data-location=./datasets/data/     --ft_data=./datasets/csv/imagenet_classID.csv     --adapter=128,128     --drop-path=0.2     --ema=0.999     --eval-scale=0.5     --grad-clip-norm=1     --ls=0.02     --mg=0.05     --eval_epoch=9     --csv-img-key=filepath     --csv-caption-key=title     --supervised-label-key=class     --exp_name=ImageNet/R-Adapter_full_finetune
```

### Retrieval Fine-Tuning on COCO

#### Preparing the COCO Dataset

```bash
ln -s /path/to/your/COCO/dataset ./datasets/data/coco
python datacreation_scripts/coco_csv_creator.py
```

#### Running the Fine-Tuning Script

```bash
torchrun --master_port 103 --nproc_per_node 4 -m src.main_retrieval     --train-dataset=coco     --epochs=10     --lr=5e-4     --wd=0     --workers=8     --batch-size=512     --model=ViT-B/16     --eval-datasets=coco,f30k     --template=openai_imagenet_template     --save=./checkpoints/     --data-location=./datasets/data/     --ft_data=./datasets/csv/coco_classID.csv     --adapter=256,256     --drop-path=0.2     --ema=0.999     --eval-scale=0.8     --ls=0     --mg=0.05     --eval_epoch=9     --csv-img-key=filepath     --csv-caption-key=title     --supervised-label-key=class     --exp_name=Retrieval/R-Adapter_retrieval_finetune
```

### Few-Shot Classification on ImageNet

#### Preparing the Few-Shot Dataset

Ensure that you have a subset of ImageNet with `k=16` samples per class.

#### Running the Few-Shot Fine-Tuning Script

```bash
torchrun --master_port 126 --nproc_per_node 4 -m src.few_shot     --train-dataset=ImageNetK     --epochs=50     --lr=5e-4     --wd=0     --workers=8     --batch-size=256     --model=ViT-B/16     --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetSketch,ImageNetA     --template=openai_imagenet_template     --save=./checkpoints/     --data-location=./datasets/data/     --ft_data=./datasets/csv/imagenet_classID.csv     --adapter=4,4     --drop-path=0.2     --ema=0.999     --eval-scale=0.5     --grad-clip-norm=1     --ls=0.02     --mg=0.05     --csv-img-key=filepath     --csv-caption-key=title     --supervised-label-key=class     --warmup_length=0     --k=16     --exp_name=ImageNet/R-Adapter_few_shot_16shot
```

### Few-Shot Base-to-Novel Fine-Tuning

We perform few-shot fine-tuning on various datasets to evaluate the adaptability of R-Adapter to new tasks, especially transitioning from base classes to novel classes.

#### Datasets

The datasets used include:

- **ILSVRC2012 (ImageNet)**
- **Caltech101**
- **Oxford Pets**
- **Stanford Cars**
- **Oxford Flowers**
- **Food101**
- **FGVC Aircraft**
- **SUN397**
- **DTD**
- **EuroSAT**
- **UCF101**

Ensure you have downloaded and prepared these datasets as per the instructions in `DATA.md`.

#### Running the Base-to-Novel Fine-Tuning Script

```bash
datasets=(ILSVRC2012 caltech-101 oxford_pets stanford_cars oxford_flowers food-101 fgvc_aircraft sun397 dtd eurosat ucf101)

for dataset in "${datasets[@]}"; do
    torchrun --master_port 126 --nproc_per_node 4 -m src.main_few_shot_novel         --train-dataset=${dataset}         --epochs=101         --lr=5e-4         --wd=0         --workers=8         --batch-size=32         --model=ViT-B/16         --template=openai_imagenet_template         --save=./checkpoints/         --data-location=./datasets/data/         --adapter=4,4         --drop-path=0.2         --ema=0.9         --eval-scale=0.5         --grad-clip-norm=1         --ls=0         --mg=0.1         --csv-img-key=filepath         --csv-caption-key=title         --supervised-label-key=class         --warmup_length=500         --k=16         --save=False         --exp_name=${dataset}/R-Adapter_few_shot_16shot
done
```
