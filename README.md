# Computer Vision for Autonomous Driving

The aim of this project is to create a Computer Vision tool used in the Autonomous Driving Industry. The tool in question is Instance Segmentation with his corresponding labeling and masking. To achieve it, we used a [Mask R-CNN]([https://arxiv.org/abs/1703.06870](https://arxiv.org/abs/1703.06870)) pre-trained in [COCO Dataset](https://cocodataset.org) (very used for object detection), we fine-tuned and did transfer learning to fit it to our use case, that uses [Cityscapes Dataset]([https://www.cityscapes-dataset.com/](https://www.cityscapes-dataset.com/)).

The repo is the Final Project delivery for the UPC [Artificial Intelligence with Deep Learning Postgraduate Course](https://www.talent.upc.edu/ing/estudis/formacio/curs/310401/postgraduate-course-artificial-intelligence-deep-learning/) 2021 edition, authored by:
- David Albiol
- Marc Martos
- Pablo Mayo
- Marc Robles

Advised by professor Laia Tarrés 

# Instance Segmentation in action

![Result_1](https://github.com/MarcKami/aidl-team6-project/blob/master/docs/exps/Experiment%201/Result_01.PNG)
  
# Content

- [How to run](#How-to-run)
- [Dataset](#Cityscapes)
- [Model](#Model)
- [Experiments](#Experiments)
- [Final Conclusions](#Final-Conclusions)

# How to run

## Requierements

To run this code you should download and install the following software:
- [Python 3.8]([https://www.python.org/downloads/release/python-380/](https://www.python.org/downloads/release/python-380/))


## Dataset Preparation

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
    3. Repeat steps `i` and `ii` for every city in the dataset

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

# Cityscapes

This project uses the well-known Cityscapes dataset, which contains a diverse set of stereo video sequences recorded in street scenes from 50 different German cities, with high quality pixel-level annotations of 5000 frames in addition to a larger set of 20000 weakly annotated frames.  In our project we only used the fine annotated images.

![CityscapesExample](https://github.com/MarcKami/aidl-team6-project/blob/master/docs/img/CityscapesExample.PNG)

Since the goal was to detect up to 8 different classes in an image, a summary of the samples for both; train and validation is given down below:

| Label/Class   | Train Sample  |
| ------------- |-------------- |
| Person        | 17.9k         |
| Rider         | 1.8k          |
| Car           | 26.9k         |
| Truck         | 0.5k          |
| Bus           | 0.4k          |
| Train         | 0.2k          |
| Cycle         | 0.7k          |
| Bicycle       | 3.7k          |


The scope of this project was to enable the detection of objects among eight different options. Nevertheless, the number of options can be customised when creating the class model. The following image shows the case of this project in which 1 + 8 classes were selected, being 1 the class related to the background. 

![NumClasses](https://github.com/MarcKami/aidl-team6-project/blob/master/docs/img/NumClasses.PNG)

## Preparation  

The decision to use Cityscapes as provider of the Dataset was based on:
  

-   Well-known predictions of the Dataset in other projects and trannings found and well reputation of the provider
    
-   Big number of images that Dataset contains
    
-   Dataset’s folder is ready to use in a project, containing a very good organization
    
-   High quality of images
    
-   Cityscapes is a dataset of street scenes suitable to develop autonomous driving applications
    
-   Most of Cityscapes classes are present in COCO, so the possibility to do Transfer Learning is real
    
  

![](https://lh6.googleusercontent.com/HKka7K_oa49lCw1v_d8WIJHaqcGTB431w9snZ1mev2T99SvXDd7XcDVy9DE4uYK9ffRkbRsdLvHy8NKTlQ25YpwvyRHcsdJ833cjiNzyz55Ohc5MD-Jrk7dQe-sP2NxoazNu9kx1)
 

Anyway, it was necessary to apply a layer of Preprocessing in order to prepare the images before using it as Input and Ground Truth in the Neural Network, everything developed in a class ready to get hyper parameters and train or validation image datasets:
	
    class  CityscapesInstanceSegmentation(Dataset)
  

Definitions done in the class are: 
  
-   Label and all meta information about the Dataset  
-   List of all possible labels defined on this project    
-   Mask list  
		
    ```python
    mask_list  = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
    ```
    
-   Take the folder path of each subset of metadata      
    ![](https://lh6.googleusercontent.com/j9ciikj2ChKyC_diSjErGaZBjpkYEMATnjB43ZbQJPdlFOe2Oio7Gbp4BAEWAx3djAvTZDiJD_kVOOKaJFLT-IrGosftBZZpHMBrKOyLN4g4JOvahheDDsn8p_9sMi00ZqYdK7Ju)

Then we should use the Masks contained in the Dataset (gtFine folder) to define each target used as outputs in this project:

-   Labels  
    ![](https://lh6.googleusercontent.com/DxZIM6lzoDbILm2NDIqmvI1aB7WnpM-0umNbT0HLo7z08ppnlTgFQ5ql15uFxiqcFvqbn3bCCELYGNvKOYsBInKGmOk8wdFodFgQRCRgFE3loGjPh8dBIOQneCDHEY59jRdfb0Vt)
    
-   Boxes  
    ![](https://lh6.googleusercontent.com/B3oB-strDx62qgLPAjM60GC_oWNdrlzg2GBjRvHah_7TMT_dGN1AbDrMe2k7wweq0AUeCtxiR9zLrzewiUYHUmBdWYchS5V9qHwedIotxt4Wbg7WjxML_h0-jMIp_laELzsvd_yF)  
    
-   Masks

    ![](https://lh4.googleusercontent.com/CoavpqhKLMac3nds4-6wP0fndB6dELUQe9dxe624MCfEhNCbhwtW73SeF9LEeSA5wQ48PGi7gqJTUas_cqmdWCeMou4FJisGIVSPEdZ5zIOktQwTpw3XuvnNcA4x_dnPx1BXkFpB)


Finally, during the training, the model expects 2 arguments so both are defined in this class:
  
-   Input tensors:  
    Input images converted to RGB and transformed into a Tensor with a defined size
    ![](https://lh3.googleusercontent.com/p7Iynb6aLhzN1H4jGSZwYiw8dM45-5OL8gSwr8ZCd-17ALzFDQw8ynat9pk2JwrmtGSM8t3sjC1yYBrenxtFYiDRQl0sZPUA4WnuV0gEafICisBkMKXJ30kC9OmIxpiwlhSKbTaA)  
  
-   Targets:  
    Output targets are defined and converted into Tensors
    ![](https://lh4.googleusercontent.com/GCsDNkxeR-5qtNAup7QUbI4-fS3_bu41Eqlq3ZKVvdIKrZJblJiUQtPwhUCndklufwQnVqFOCgjFX1AzVb1D4S63zax2sbdLQ440a9eSxM2Vu7PDn9xcRunNYk8Sb0Zd_P9YaZXf)


# Model

![](https://github.com/MarcKami/aidl-team6-project/blob/master/docs/img/Heads.png)
	
Our model is a pre-trained Mask-RCNN in COCO Dataset but adapted to our use case. So, to do that, we changed the two last heads of this model to fit it with Cityscapes Dataset.
The implementation has been change the num of classes of the box predictor head and also have the possibility to change the hidden layers and the random range size of the input images.
```python
model = detection.maskrcnn_resnet50_fpn(pretrained=pretrained, min_size=min_size, max_size=max_size)
    
# get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(
	in_features,
	num_classes)

# now get the number of input features for the mask classifier
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
# and replace the mask predictor with a new one
model.roi_heads.mask_predictor = nn.Sequential(OrderedDict([
	    ("conv5_mask", nn.ConvTranspose2d(in_channels=in_features_mask, out_channels=hidden_layer, kernel_size=2, stride=2)),
	    ("relu", nn.ReLU(inplace=True)),
	    ("mask_fcn_logits", nn.Conv2d(in_channels= hidden_layer, out_channels=num_classes, kernel_size=1))]))

```
## Pipeline
Here we can see the big picture of the whole process that has our pipeline from the dataset adquisition to prediction rendering:
- [Dataset](#Cityscapes)
- [Dataset Preparation](#Preparation)
- [Modified Mask-RCNN](#Modified-Mask-RCNN)
- [Parameters & Optimizer](#Parameters-and-Optimizer)
- [Train](#Train)
- [Output](#Output)

	![](https://github.com/MarcKami/aidl-team6-project/blob/master/docs/img/Architecture.PNG)
 
### Load Dataset
 
First step is load Train and Validation Dataset provided from Cityscapes and using the Dataset Class defined:
  
![](https://lh6.googleusercontent.com/Ht9SgwSmIb0SHS9VdWxptui0fMSabKe1dmpP72hIuc2F2jeifOtGo34U11G6zVOxTGeYVmnakOXaaNqTB7Kzp0UwcEiM1fNMtyQT2DRbvCWRL0DCaqWXMtRh64Thv8coHgCg_fUk)

In order to train faster, images have been loaded in memory and then loaded train and validation dataset:

![](https://lh5.googleusercontent.com/lIgDxQn5YyH5rXycdS7xDcAWMtg5buNLvPSSjVaX_OapnqryEJO2_Dpu1xo6EiOYAeUHJBRsYUOz1OzxhTPsphLyW3cEAqbIsabWda0SLHaC9ytbhqXI0wRPcDCnBOxcw3LQSO-n)

### Modified Mask-RCNN 

Next step is the model definition, based on a Mask-RCNN: 

![](https://lh3.googleusercontent.com/kiSuLV6RH56iIuMLjOs3j3FkeRyvzKAfyhPd-7Vdc7RMXDn8zw334LcN-Z3La7b1zPDdYmrlVv4C7M7Wud9fp8pPrALLyck5aSNOMCO3HA3MlRKtlw11KVm_uIM-ZtKyYkfUiOhP)
  
In the definition of the Model, a pretrained Mask-RCNN model has been used based on Resnet50:

![](https://lh5.googleusercontent.com/hdQ7LQzXu5hHOhecHTwMlR3XCgYcSSVtr4d-t0iMnlWWKFnZA9cCOrVO-kWftfCMyExxxufbTq46e88CHXlcvHSWFuxBysvo4QxQOhVY-TxyMRFpMiIaSVNnjeBsnl6bxFJ5WhGG) 

Afterwards, next components of the pretrained Mask-RCNN have been replaced:

-   Box Predictor Head
    
-   Mask Predictor Head
    
![](https://lh4.googleusercontent.com/5m0jNIkRmHFEaHDYyEG7X3mRmnWAb4lvaKO0vUuewOUaH0ZOVTHzYPspBjtIAjUJseE4TDCugdwHXIcibilz6bt_f-VLOZA_zbUyUZeaqEVFJO9meugcuh53tBFqwI-JTgi2IT3M)  

### Parameters and Optimizer

Parameters label are defined using the new Fast-CRNN Head Mask and Box predictor and Adam Optimizer has been used in this model: 

  ![](https://lh3.googleusercontent.com/EE7zKfJ5XBUMyblhoe3-MbDKtksFToCLlCft59i_8Z6qhiFPTMPuqW9PKLLkK9vcIrQRBqdCSDq3insOWHF0__doWZimGcEMQdwCUv0C1HWchpvlsvMKsZOk7XDo3pBWqDt_I2Py)

### Train


### Output  

Our head applied on top of the pre-trained Mask-RCNN gives three different kind of outputs: 
Bounding Box: Box that surrounds the object to identify it. 
Label: Result of the classification processes between our 8 classes that classifies the object inside the Bounding Box.
Mask: Polygonal mask that has the same shape as the object labeled inside the bounding box.

We’ve applied a visualizers to see our results that renders Label + Bounding Box + Mask combined with different colors per each instance. Should be remarked that in this visualization, we're showing only the objects detected with **at least 80% of confidence**, but if you want to modify it only need to change `scores_threshold = 0.8` inside `utils.py`:

![](https://github.com/MarcKami/aidl-team6-project/blob/master/docs/img/Output.PNG)

# Experiments
- [Experiment 1](#Experiment-1)
- [Experiment 2](#Experiment-2)
- [Experiment 3](#Experiment-3)
## Experiment 1
Pre-Trained model on COCO without fine-tuning
### Hypothesis
We expect quality predictions because the Cityscapes dataset exhibits a high overlap with the COCO dataset. In particular, from the eight classes we want to predict in Cityscapes, seven of them are also present in COCO (person, car, bicycle, motorcycle, truck, bus and train) while only one (rider) is missing. 
### Setup
We prepared a set of 6 photos to check the visual results of the inference using this model.
### Results
As we expected the pretrained MaskRCNN was able to accurately detect and segmentate the single instances of the above mentioned common classes.  But we noticed that the class `rider`, present in Cityscapes and relevant for autonomous driving use case, is not present in the predicted results not even in the COCO datasat.
![Result_00](https://github.com/MarcKami/aidl-team6-project/blob/master/docs/exps/Experiment%201/Result_00.PNG)
![Result_06](https://github.com/MarcKami/aidl-team6-project/blob/master/docs/exps/Experiment%201/Result_06.PNG)

### Conclusions
The results confirm our intuition that the model without fine-tuning constitutes a very solid baseline model for the classes present in both datasets.

To make predictions about the class “rider”, which is not found in [COCO labels](https://gist.github.com/iitzco/3b2ee634a12f154be6e840308abfcab5), we proceed to update the detection and segmentation heads of the MaskRCNN by randomly initializing the last layers and reducing to 8 (our subset of Cityscapes) plus one additional class for the background.  

## Experiment 2
1000 Samples over 20  epochs with (800, 1024) range random size & lr= 0.001 (default Adam)
### Hypothesis
The updated model should recognise the new class “rider” and distinguish it from "person2in the validation images.  To start our fine-tuning we opt for a simplwith a standard learning rate of 0.001
### Setup
In this case we prepared a set of 1000 training images and 200 validation images. As a data augmentation technique, we used random image resizing from 800 and 1024 pixels (short side) as in the MaskRCNN Paper (https://arxiv.org/abs/1703.06870). And we used a vanilla version of Adam optimizer with a learning rate of 0.001.
### Results
After 4-5 epochs the learning curves stagnate for both training and validation sets, suggesting some kind of learning rate decay policy may be necessary to observe a further progression.  As a consequence of that stagnation we cannot see all the desired labels, at least with the appropriate confidence, in the predictions. We're able to see the rider class well classified as we wanted but cannot see other relevant classes like person.
- Total Loss:

![](https://github.com/MarcKami/aidl-team6-project/blob/master/docs/exps/Experiment%202/TotalLoss.PNG)
- Splited Losses:

![](https://github.com/MarcKami/aidl-team6-project/blob/master/docs/exps/Experiment%202/SplitLoss.PNG)
- Visual Results:

![](https://github.com/MarcKami/aidl-team6-project/blob/master/docs/exps/Experiment%202/Result_00.PNG)
![](https://github.com/MarcKami/aidl-team6-project/blob/master/docs/exps/Experiment%202/Result_06.PNG)

### Conclusions
Given the results, we've able to include the raider class as we want, but other relevant classes are missing. So we think that it's caused by the stagnation of the learning rate in the early stage of the training process. Due this situation, we think that the vanilla Adam is not enough to perform this task, so we'll need an extra experiment to optimize it.

## Experiment 3
1000 Samples over 10 epochs with (800, 1024) range random size & lr= 0.001 **using ReduceLROnPlateau scheduler**
### Hypotesis
Given the good results on the first steps of the previous experiment and the stuck on the learning of the model, we expect to solve this problem and have better results implementing an [ReduceLROnPlateau](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html) scheduler and update the learning rate accordingly. This should resolve the classification problem that we have with person class and other classes missing in the experiment 2 predictions
### Setup
In this case we used the same set of 1000 training images and 200 validation images. This time we used the same learning rate given the results in the previous experiment, but in the other hand, as a result of lr stuck, we decide to implement the scheduler [ReduceLROnPlateau](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)
### Results
TODO
### Conclusions
TODO

# Final Conclusions
- Data preprocessing can be a time consuming task.
- Pre-implemented tools (typically Github repos) may require major adaptations to be integrated with your dataset/ model implementation.
- How to preprocess different aspects of a dataset to fit your model/approach.
- Memory management & optimization can (very) significantly speed up training
- Transfer learning is a very powerful technique: very good results even without fine-tuning if datasets are similar.

