# GausSat: Gaussian Splatting for remote sensing Super-Resolution

This is the repository for my Computer Science bachelor's degree final thesis. It is based on the work done by [Chen et al., 2025](https://arxiv.org/abs/2501.06838) and expands their architecture for multispectral images.

## Structure

* **checkpoints/:** Training checkpoints of the GausSat model.

* **data/:** Data classes for the different datasets used. The Data section indicates the dataset structure each dataset expects and how to prepare them. This folder also stores some transformations for data augmentation and the different metrics used in the training and testing process.

* **docs/:** All documents generated during the development of this project. The ```old_reports/``` subfolder has the previous reports created during the development, ```figures/``` has all figures included in the final report, and ```gantt_diagram/``` has the initial gantt_diagram created.

* **experiments/:** Train and test configuration files in .yaml format for the different experiments conducted in the project. New files can be created to generate different configurations and results, or the existing ones can be modified and used as template.

* **logs/:** Tensorboard logs generated during training and testing of the model.

* **models/:** Models library with the code for every model used in this project. Every model has each construction block separated into different python files. GausSat is the main NN used, EDSR is the backbone we used for our experiments, and HFN is the NN used to upsample Sentinel-2 bands from 20 m/px and 60 m/px to 10 m/px.

* **notebooks/:** Spontaneous notebooks used during the development for specific tasks. They are collected here for the sake of organization, but if you need to use any of them, I advice you move them to the root folder of this repository to avoid possible conflicts with python paths.

* **results/:** Results from test runs in .csv format.

* **submodules/:** C++/CUDA files for the 2D rasterizer of GausSat.

* **weights/:** Trained weights for every model used and different experiments. It has an [EDSR](https://arxiv.org/abs/1707.02921) baseline extracted directly from the official source, the [HFN](https://www.sciencedirect.com/science/article/abs/pii/S0924271622003331) weights trained by us, and the GausSat weights for each experiment we did.

* **ms_preparation.py:** Script to preprocess the Sentinel-2 dataset before feeding it to GausSat.

* **train.py/test.py:** Training and testing scripts for GausSat.

## Installation

1. Clone the repository using:  
 ```bash
 git clone https://github.com/adriangt2001/TFG-Satellite-GSSR.git
 ```
2. Install dependencies with pip. Remember to use an enclosed Python environment to install dependencies:
 ```bash
 cd TFG-Satellite-GSSR
 python -m venv .venv # Create an environment
 . ./venv/bin/activate # Activate the environment
 pip install -r requirements.txt # Install dependencies
 ```
3. Install Pytorch and CUDA. We tested this with PyTorch 2.5.1, but most importantly, make sure to install PyTorch with the same verison as the CUDA Toolkit used. In our case we used CUDA version 12.4. In the official [PyTorch](https://pytorch.org/get-started/locally/) and [CUDA](https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local) websites.  
3.1. Make sure to check your maximum supported CUDA version before installing using ```nvidia-smi``` command. If you want to install a different version than 12.4, you can find it easily with a quick search. Just be careful because for version 12.1 the documentation is wrong and it installs CUDA instead of the CUDA Toolkit, which can cause some problems.  
3.2. In the case of PyTorch, it can be easily installed with the following command, just change the 124 part of the end for your specific CUDA version:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

4. Compile the CUDA submodules inside the ```submodules/``` folder:
```bash
cd submodules/diff-srgaussian-rasterization
pip install .
```

5. Now everything should be up and ready.

## Data

This project uses 2 different datasets to train the models.

### DIV2K

[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) is a well known dataset used for the Super-Resolution task. Preparing it is as easy as downloading it from the official page. In a folder named ```DIV2K```, download the DIV2K HR train data and the DIV2K HR validation data. Once they are downloaded, they are ready. ```data/dataset_div2k.py``` is the script that contains our custom dataset class for this dataset.

### Sentinel-2

Sentinel-2 is a satellite that captures multispectral information from 12 different parts of the electromagnetic spectrum. To use our same dataset, go to the official [Copernicus Browser](https://browser.dataspace.copernicus.eu/) website and download the following images:
* S2B_MSIL2A_20220629T100029_N0510_R122_T35VML_20240630T200504
* S2B_MSIL2A_20241127T104309_N0511_R008_T31TDG_20241127T131852
* S2B_MSIL2A_20241130T105329_N0511_R051_T30TXK_20241130T132639
* S2B_MSIL2A_20241130T105329_N0511_R051_T31TCH_20241130T132639
* S2C_MSIL2A_20250516T014601_N0511_R074_T52KBG_20250516T045613

We used the first three as training set, the fourth as validation and the last one as test. To preprocess the data, we used the HFN network with the pretrained weights ```HFN_20.pt``` and ```HFN_60.pt``` in the ```weights/``` folder. To preprocess the dataset, just run the ```ms_preparation.py``` script with the following arguments:
```bash
python ms_preparation.py <dataset_path> <weights20_path> <weights60_path>
```

The data at first is expected to be in the following folder structure:
```
Sentinel-2/
├── train/
│   ├── Raster1/
│   │   ├── R10m/
│   │   │   └── Multiple bands from this resolution separated
│   │   │       in different files.jp2
│   │   ├── R20m/
│   │   │   └── Multiple bands from this resolution separated
│   │   │       in different files.jp2
│   │   └── R60m/
│   │       └── Multiple bands from this resolution separated
│   │           in different files.jp2
│   ├── Raster2/
│   │    └── ...
│   └── ...  
├── valid/
│   ├── Raster1/
│   │   └── ... 
│   └── ...  
└── test/
    └── ...
```

For the preprocessed data, it will generate a subfolder inside the ```train/```, ```valid/``` and ```test/``` folders, and will store all generated images as tensors in a .pt format.

```data/dataset_sen2.py``` has the dataset class used to preprocess the Sentinel-2 data, while ```data/dataset_sen2_processed.py``` implements the dataset class that uses the preprocessed Sentinel-2 data.

## Training
To train the GausSat model, just run the ```train.py``` script. You can run the script writing each argument manually in every call or use a configuration file from ```experiments/``` like this:
```bash
python train.py --config experiments/experiment_train1.yaml
```
Make sure that the configuration file has the correct dataset path.

## Test
To train the GausSat model, just run the ```test.py``` script. You can run the script writing each argument manually in every call or use a configuration file from ```experiments/``` like this:
```bash
python test.py --config experiments/experiment_test1.yaml
```
Make sure that the configuration file has the correct dataset path.