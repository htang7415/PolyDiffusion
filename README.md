
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

torchrun --nproc_per_node 1 scripts/train.py \
    hydra.run.dir=ckpt/polymer_test \
    wandb.name=null \
    data=/home/htang228/Machine_learning/Diffusion_model/PolyDiffusion/data/water_soluble_polymer_safe.txt
```

${input_path} is the path to the dataset file with a SMILES in each row.
${data_path} is the path of the processed dataset.

Prompt:

we have data: polymer repeat unit SMILES (with two * of each, different to drug) in PI1M.txt, which can be used for polymer training data.

give strategy for polymer repeat unit design using safe and genmol (diffusion model) in this program:
1. use polymer data for training. consider how to fragmentation by safe for polymer repeat unit.
2. after sampling, we hope the generated polymer repeat unit will have two * in generated_samples.txt by using denovo.py without post process.
3. after sampling, we will conider the uniquiness, validity, diversity, quality of these generated polymer repeat units.

give instructions step by step and tell me where to change of the code.