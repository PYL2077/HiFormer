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
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge

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
To train GSRFormer on a single node with 4 GPUs for 40 epochs, run:
```bash
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
           --backbone resnet50 --batch_size 16 --dataset_file swig \
		   --encoder_epochs 20 --decoder_epochs 0 \
           --num_workers 4 --num_enc_layers 6 --num_dec_layers 5 \
            --dropout 0.15 --hidden_dim 512 --output_dir GSRFormer
```

- A single epoch takes about 38 minutes. Training CoFormer for 40 epochs takes around 25 hours on a single machine with 4 RTX 3090 GPUs.          
- We use AdamW optimizer with learning rate 10<sup>-4</sup> (10<sup>-5</sup> for backbone), weight decay 10<sup>-4</sup> and β = (0.9, 0.999).    
    - Those learning rates are divided by 10 at epoch 30.
- Random Color Jittering, Random Gray Scaling, Random Scaling and Random Horizontal Flipping are used for augmentation.

## Evaluation
```bash
python main.py --saved_model GSRFormer/checkpoint.pth --output_dir GSRFormer --dev
```

```bash
python main.py --saved_model GSRFormer/checkpoint.pth --output_dir GSRFormer --test
```

## Inference
To run an inference on a custom image, run:
```bash
python inference.py --image_path inference/filename.jpg \
                    --saved_model CoFormer_checkpoint.pth \
                    --output_dir inference
```

