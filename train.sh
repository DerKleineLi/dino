python -m torch.distributed.launch --nproc_per_node=1 main_dino.py --warmup_epochs 0 --arch vit_tiny --epochs 5 --out_dim 8192 --data_path /mnt/raid/hli/datasets/imagenet/ilsvrc2012/train --output_dir /mnt/raid/hli/models/dino_vanilla