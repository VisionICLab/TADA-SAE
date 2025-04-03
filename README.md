# TADA-SAE
A method for detecting breast cancer from thermal images using bilateral symmetry and learned texture features. Featured in ISBI 2025: https://biomedicalimaging.org/2025/full-program/


## Dataset
The dataset used in this project can be found here:
https://www.kaggle.com/datasets/tiennelescarbeault/dmr-ir-breast-cancer-detection-for-tada-sae/data

It comprises the formatted breast segmentation masks for both normal and anomalous cases and train-test splits. The original dataset can be found here: http://visual.ic.uff.br/en/proeng/thiagoelias/

### About
It comprises of thermal images from 57 patients, in which 36 were diagnosed with breast cancer. In a thermal image, this is characterised by a temperature increase over the affected side. In this version, left and right breasts were extracted separately, which is useful to compare their asymmetries, thus indicating possible breast cancer.

## Usage
To reproduce the experiments found in the paper, 3 files are provided: ```run_dmrir_baselines.py```, ```run_dmrir_ablation.py``` and ```run_dmrir_tadasae.py```. 

### Baseline experiments
The file ```run_dmrir_baselines.py``` contains the code to reproduce the baseline experiments, which are supervised classification using ResNet-18, InceptionV3 and a simple CNN. The parameters can be adjusted in the command line or in the ```configs/supervised_dmrir.yaml``` file.

**Example**: ```python run_dmrir_baselines.py --experiment resnet_18```

### Ablation experiments
The file ```run_tadasae_ablation.py``` contains the code to reproduce the ablation experiments, which are the experiments to evaluate the importance of each component of the TADA-SAE method. The parameters can be adjusted in the command line or in the ```configs/ae_dmrir.yaml``` or ```configs/tadasae_dmrir.yaml``` files, depending on the chosen ablation experiment. Three (3) ablation experiments are provided:

- **ae_full_im**: Simple autoencoder to extract image features from the whole image, followed by a classification. Therefore, no texture disentanglement or left-right symmetry is used.

- **ae_left_right**: Simple autoencoder to extract image features from the left and right images, followed by a classification. Therefore, no texture disentanglement is used but does left-right comparisons.

- **lsae_full_im**: A swapping autoencoder based on LSAE (Zhou et al., 2022) to extract image features from the whole image, followed by a classification. Texture disentanglement is performed but no left-right comparison is conducted.

**Example**: ```python run_dmrir_ablation.py --experiment ae_full_im```

### TADA-SAE
The file ```run_dmrir_tadasae.py``` contains the code to reproduce the TADA-SAE method. The parameters can be adjusted in the command line or in the ```configs/tadasae_dmrir.yaml``` file. The TADA-SAE method is a combination of the three components: texture disentanglement, left-right symmetry and classification. It is based on the LSAE (Zhou et al., 2022) autoencoder, which is used to extract image features from the left and right images, followed by a classification. For this model, weights are provided in the weights folder.

**Examples**: 

Use pretrained:
```python ./run_dmrir_tadasae.py --test-only --checkpoint <path_to_weights>.pth```

Train from scratch:
```python ./run_dmrir_tadasae.py ```
