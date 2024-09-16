# KSSANet

Code of KSSANet: "KAN-Driven Spatial-Spectral Attention Networks for Hyperspectral Image Super-Resolution"

Due to the limitations of physical imaging, acquir-
ing high-resolution hyperspectral images (HR-HSIs) has always
been a significant challenge. Single hyperspectral image super-
resolution (SHSR) technology aims to generate corresponding
HR-HSIs by processing low-resolution hyperspectral images (LR-
HSIs). Compared to multi-source data fusion methods, SHSR
relies solely on a single low-resolution image and does not
require additional auxiliary information or multimodal data,
making it more flexible and efficient in data acquisition. Recently,
Kolmogorov–Arnold Networks (KAN), which derive from the
Kolmogorov–Arnold representation theorem, show great poten-
tial in modeling long-range dependencies. In this paper, we
further investigate the potential of KAN for hyperspectral image
restoration. Specifically, we propose a spatial-spectral attention
block (SSAB) module, which includes a KAN-based spatial
attention module (KAN-SpaAB) and a KAN-based spectral
attention module (KAN-SpeAB), designed for the restoration
of spatial and spectral information, respectively. Experimental
results demonstrate that KSSANet outperforms existing methods
in both quantitative evaluation and image generation quality,
achieving state-of-the-art (SOTA) performance. 

## Requirements
- Python 3.8+
- PyTorch 1.4+
- CUDA 10.1+
- torchvision 0.5+
- h5py 2.10+
- matplotlib 3.2+

## Datasets
- CAVE: https://www.cs.columbia.edu/CAVE/databases/multispectral/
- Chikusei: https://naotoyokoya.com/Download.html 
- Pavia: https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_Centre_and_University

## Usage
```
python train.py --dataset CAVE --batch_size 32 
```

## Acknowledgement
Our code references [efficient-kan](https://github.com/Blealtan/efficient-kan.git) and [pykan](https://github.com/KindXiaoming/pykan.git). Thanks for their greak work!