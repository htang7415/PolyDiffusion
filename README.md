
# Training with Polymer Dataset
-------
## First, convert your SMILES dataset into SAFE by running the following command:
```bash
python genmol/scripts/preprocess_polymer_safe.py \
    data/water_soluble.txt \
    data/water_soluble_polymer_safe.txt \
    --num_attachments 2
```

## Second, train model.
```bash
# Train with reduced batch size to fit in available GPU memory
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 scripts/train.py \
    hydra.run.dir=ckpt/polymer_model \
    wandb.name=null \
    data=/home/htang228/Machine_learning/Diffusion_model/PolyDiffusion/data/water_soluble_polymer_safe.txt \
    loader.global_batch_size=32 \
    trainer.max_steps=5000 \
    callback.dirpath=./checkpoints_polymer
```

```bash
torchrun --nproc_per_node 1 scripts/train.py \
    hydra.run.dir=ckpt/polymer_test \
    wandb.name=null \
    data=/home/htang228/Machine_learning/Diffusion_model/PolyDiffusion/data/water_soluble_polymer_safe.txt
```

## Third, generate polymers

```bash
python scripts/exps/denovo_polymer.py
```

```bash
${input_path} is the path to the dataset file with a SMILES in each row.
${data_path} is the path of the processed dataset.
```
