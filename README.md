# DE-NAF

The code for our work [DE-NAF: Decoupled Neural Attenuation Fields for Sparse-View CBCT Reconstruction].

## 1.Setup

We recommend using [Conda](https://docs.conda.io/en/latest/miniconda.html) to set up an environment.

``` sh
# Create environment
conda create -n naf python=3.9
conda activate naf

# Install pytorch (hash encoder requires CUDA v11.3)
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install other packages
pip install -r requirements.txt
```

## 2.Dataset

Download eight CT datasets from [here](https://drive.google.com/drive/folders/1-Qzp5Rajx8gZIGqUkpOGYyAf8O6QAZh2?usp=drive_link). Put them into the `./data` folder.

You also can make your own simulation dataset with TIGRE toolbox. Please first install [TIGRE](https://github.com/CERN/TIGRE/blob/master/Frontispiece/python_installation.md).

Put your CT data in the format as follows.

```sh
├── raw                                                                                                       
│   ├── XXX (your CT name)
│   │   └── img.mat (CT data)
│   │   └── config.yml (Information about CT data and the geometry setting of CT scanner)
```

Then use TIGRE to generate simulated X-ray projections.

``` sh
python dataGenerator/generateData.py --ctName XXX --outputName XXX_50
```

## 3.Training and evaluation
Experiments settings are stored in `./config` folder.

For example, train DE-NAF with `chest` dataset:

``` sh
python train.py --config ./config/chest.txt
```
The evaluation outputs will be saved in `./log` folder.

## Citation

Our work is about to be published. Please cite our work if you find this repository helpful to your project.

## Acknowledgement
* [NAF](https://github.com/Ruyi-Zha/naf_cbct.git), [TensoRF](https://github.com/apchenstu/TensoRF.git) and [S3IM](https://github.com/Madaoer/S3IM-Neural-Fields.git) inspired our work.
* The datasets are processed through [TIGRE toolbox](https://github.com/CERN/TIGRE.git).
