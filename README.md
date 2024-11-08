## Uni MS-PS: a Multi-Scale Encoder Decoder Transformer for Universal Photometric Stereo



**Author:** Clément Hardy, Yvain Quéau, David Tschumperlé (Normandie
Univ, UNICAEN, CNRS, ENSICAEN, GREYC laboratory, Caen, France)

**Project site:**
https://clement-hardy.github.io/Uni-MS-PS/index.html

**Abstract:**
Photometric Stereo (PS) addresses the challenge of reconstructing a three-dimensional (3D) repre-
sentation of an object by estimating the 3D normals at all points on the object’s surface. This is
achieved through the analysis of at least three photographs, all taken from the same viewpoint but
with distinct lighting conditions. This paper introduces a novel approach for Universal PS, i.e.,
when both the active lighting conditions and the ambient illumination are unknown. Our method
employs a multi-scale encoder-decoder architecture based on Transformers that allows to accom-
modates images of any resolutions as well as varying number of input images. We are able to
scale up to very high resolution images like 6000 pixels by 8000 pixels without losing perfor-
mance and maintaining a decent memory footprint. Moreover, experiments on publicly available
datasets establish that our proposed architecture improves the accuracy of the estimated normal
field by a significant factor compared to state-of-the-art methods.


### Install Dependencies
The following python package are necessry to run the test:
- Python 3
- PyTorch
- OpenCV
- Pillow
- SciPy
- numpy
- timm

The code has been run with the following version:
- Python 3==3.8.11
- PyTorch==1.12.1
- OpenCV==4.8.0.74
- Pillow==9.0.1
- SciPy==1.7.1
- numpy==1.22.4
- timm=0.6.7

dvd(2024/11/06): For cluster
```shell
qlogin -q gpu # Activate GPU
module load cuda/12.4.1
module load pytorch/1.11.0/cuda/11.3.1/gpu #pytorch/1.12.1/gpu
module load opencv/4.5.4/gcc/11.2.0/module-x5zx4hp

module load python/3.11/anaconda # Check which version with module avail python
python -m venv .venv # Ignore if you already have a virtual environment
source .venv/bin/activate # Activate virtual environment

# Install packages
/work/imvia/de1450bo/repos/Uni-MS-PS-fork/.venv/bin/python -m pip install --upgrade pip
/work/imvia/de1450bo/repos/Uni-MS-PS-fork/.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/
/work/imvia/de1450bo/repos/Uni-MS-PS-fork/.venv/bin/python -m pip install opencv-python-headless pillow scipy numpy timm

# Running the Test:
/work/imvia/de1450bo/repos/Uni-MS-PS-fork/.venv/bin/python inference_file.py --path_obj /work/imvia/de1450bo/repos/Uni-MS-PS-fork/data/bearPNG --nb_img 6 --cuda
```

## Pretrained Models
Weights of the network are available at: https://drive.google.com/file/d/1scrTLjQCz0yJPecBMaaCqto2gDFrcCkE/view?usp=sharing for the Universal version and https://drive.google.com/file/d/1rM1Z0gVtDZvy6g0X9Jhc9rrHmd4tsP9F/view?usp=sharing for the calibrated version.
Place them in the weights folder.

## Running the Test
To run the inference on a single object, execute `inference_file.py` with the following command:

```
python inference_file.py --path_obj PATH_OF_YOUR_FOLDER --nb_img NB_IMG_TO_USE
```
add ``` --cuda``` if you want to use cuda\
By default, the results are place in the folder inference, this could be change by using the argument
```
--folder_save SAVE_PATH
```

The ```--calibrated``` option allows the use of the calibrated version of the neural network.

The images could be in png, jpg or in TIFF format, a mask of the object could be also provided, it should be place on the images's folder name as mask.png. 

For the calibrated version, a light_directions.txt should be provided, the format of this file should be the same as the ones in the DiLiGenT dataset.
The names of images should be numbers, for example, 1.png is associated with the first row in the light_directions.txt file, 2.png with the second row, and so on.

#### Example on DiLiGenT
Download the DiLiGenT dataset here: https://sites.google.com/site/photometricstereodata/single , extract it and then run for all objects:

```
python inference_folder.py --path_obj YOUR_PATH/DiLiGenT/pmsData --nb_img 6 --cuda
```

or for a single object

```
python inference_file.py --path_obj YOUR_PATH/DiLiGenT/pmsData/ballPNG --nb_img 6 --cuda
```


