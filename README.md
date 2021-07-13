# Computer Vision for Autonomous Driving

The aim of this project is to create a Computer Vision tool used in the Autonomous Driving Industry. The tool in question is Instance Segmentation with his corresponding labeling and masking. To achieve it, we used a [Mask R-CNN]([https://arxiv.org/abs/1703.06870](https://arxiv.org/abs/1703.06870)) pre-trained in [COCO Dataset](https://cocodataset.org) (very used for object detection), we fine-tuned and did transfer learning to fit it to our use case, that uses [Cityscapes Dataset]([https://www.cityscapes-dataset.com/](https://www.cityscapes-dataset.com/)).

The repo is the Final Project delivery for the UPC [Artificial Intelligence with Deep Learning Postgraduate Course](https://www.talent.upc.edu/ing/estudis/formacio/curs/310401/postgraduate-course-artificial-intelligence-deep-learning/) 2021 edition, authored by:
- David Albiol
- Marc Martos
- Pablo Sabino
- Marc Robles

Advised by professor Laia Tarrés 

# Instance Segmentation in action

*** SHOW IMAGES OF RESULTS ***
  
# Content

- [How to run](#How-to-run)
- [Dataset](#Dataset)
- [Architecture](#Architecture)
- [Experiments](#Experiments)

# How to run

## Requierements

To run this code you should download and install the following software:
- [Python 3.8]([https://www.python.org/downloads/release/python-380/](https://www.python.org/downloads/release/python-380/))


## Dataset

1.  Download [Cityscapes Dataset]([https://www.cityscapes-dataset.com/](https://www.cityscapes-dataset.com/)) data `gtFine_trainvaltest.zip and leftImg8bit_trainvaltest.zip` and extract them.
We just need the json files, so extract them into 'Json_files/'. Run scripts/process_dataset.py.
And the folder structure would then be (where folders under 'gtFine/' are empty):
```
data/
|-- leftImg8bit/
| |-- train/<br>
| |-- val/<br>
| |-- test/<br>
|-- gtFine/
| |-- train/<br>
| |-- val/<br>
|-- Json_files/
| |-- train/<br>
| |-- val/<br>
| |-- test/
```

2. Rename the filenames, e.g. 'aachen_000000_000019_leftImg8bit.png' --> 'aachen_000000_000019.png'. 
    1. Go to `aux_scripts/process_filename.py` and change path on line 7:
`path = './leftImg8bit/train/aachen/'`
    2. Run `python aux_scripts/process_filename.py`
    3. Repeat steps 2.1 and 2.2 for every city in the dataset

3. Define the instance classes. Use `cityscapesScripts/cityscapesscripts/helpers/label.py` to modify the trainIds as suitable for your model. Assign the label 255 to classes that you want to ignore during the training.

![labels](https://github.com/MarcKami/aidl-team6-project/blob/master/docs/img/Labels.PNG)

4. Draw the ground truth masks for the training and validation set using:
`cityscapesScripts/cityscapesscripts/preparation/json2labelImg.py`  

![json_dir](https://github.com/MarcKami/aidl-team6-project/blob/master/docs/img/JsonDir.PNG)

![mask_list](https://github.com/MarcKami/aidl-team6-project/blob/master/docs/img/MaskList.PNG)

## Installation

1. Open a Terminal to the source folder of this repo.

2. We recommend using a virtual environment, but this step is **optional**.
`python -m venv /path/to/new/virtual/environment`

3. Activate this enviroment in case that you've created one.
Win: `venv_path/Scripts/activate`
iOS: `source venv_path/bin/activate`
4. Install all the requiered libraries.
`pip install -r requirements.txt`
5. Train
`python main.py`  

### Inference
The model states are going to be saved inside models folder. To do inference you should pick one of them (the best better) and rename as `model_good` and run `inference.py` script.

**Graphical Results**

To see graphical results or also check the whole process, we used [Visual Studio Code](https://code.visualstudio.com/download) with Python and Jupyter packages. Then you can run the code running `main.py` or `inference.py` file running it in an interactive window.

![VSCode](https://github.com/MarcKami/aidl-team6-project/blob/master/docs/img/VSCode.PNG)

# Experiments
- [Experiment 1](#Experiment-1)
- [Experiment 2](#Experiment-2)
- [Experiment 3](#Experiment-3)
- [Experiment 4](#Experiment-4)
- [Experiment 5](#Experiment-5)
## Experiment 1
### Hypotesis
### Setup
### Results
### Conclusions
## Experiment 2
### Hypotesis
### Setup
### Results
### Conclusions
## Experiment 3
### Hypotesis
### Setup
### Results
### Conclusions
## Experiment 4
### Hypotesis
### Setup
### Results
### Conclusions
## Experiment 5
### Hypotesis
### Setup
### Results
### Conclusions
