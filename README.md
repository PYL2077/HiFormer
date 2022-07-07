Based on https://github.com/jhcho99/CoFormer

## Environment Setup
We provide instructions for environment setup.
```bash
# Clone this repository and navigate into the repository
git clone https://github.com/PYL2077/GSRFormer.git
cd GSRFormer

# Create a conda environment, activate the environment and install PyTorch via conda
conda create --name GSRFormer python=3.9              
conda activate GSRFormer
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge 

# Install requirements via pip
pip install -r requirements.txt                   
```

## SWiG Dataset
Annotations are given in JSON format, and annotation files are under "SWiG/SWiG_jsons/" directory. Images can be downloaded [here](https://swig-data-weights.s3.us-east-2.amazonaws.com/images_512.zip). Please download the images and store them in "SWiG/images_512/" directory.

#### Additional Details
- All images should be under "SWiG/images_512/" directory.
- train.json file is for train set.
- dev.json file is for development set.
- test.json file is for test set.

## Training
To train CoFormer on a single node with 4 GPUs for 40 epochs, run:
```bash
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
           --backbone resnet50 --batch_size 16 --dataset_file swig --epochs 40 \
           --num_workers 4 --num_glance_enc_layers 3 --num_gaze_s1_dec_layers 3 \
           --num_gaze_s1_enc_layers 3 --num_gaze_s2_dec_layers 3 --dropout 0.15 --hidden_dim 512 \
           --output_dir CoFormer
```
To train CoFormer on a Slurm cluster with submitit using 4 RTX 3090 GPUs for 40 epochs, run:
```bash
python run_with_submitit.py --ngpus 4 --nodes 1 --job_dir CoFormer \
        --backbone resnet50 --batch_size 16 --dataset_file swig --epochs 40 \
        --num_workers 4 --num_glance_enc_layers 3 --num_gaze_s1_dec_layers 3 \
        --num_gaze_s1_enc_layers 3 --num_gaze_s2_dec_layers 3 --dropout 0.15 --hidden_dim 512 \
        --partition rtx3090
```

- A single epoch takes about 45 minutes. Training CoFormer for 40 epochs takes around 30 hours on a single machine with 4 RTX 3090 GPUs.          
- We use AdamW optimizer with learning rate 10<sup>-4</sup> (10<sup>-5</sup> for backbone), weight decay 10<sup>-4</sup> and β = (0.9, 0.999).    
    - Those learning rates are divided by 10 at epoch 30.
- Random Color Jittering, Random Gray Scaling, Random Scaling and Random Horizontal Flipping are used for augmentation.

## Inference
To run an inference on a custom image, run:
```bash
python inference.py --image_path inference/filename.jpg \
                    --saved_model CoFormer_checkpoint.pth \
                    --output_dir inference
```

