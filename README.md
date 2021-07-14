# Computer Vision for Autonomous Driving

The aim of this project is to create a Computer Vision tool used in the Autonomous Driving Industry. The tool in question is Instance Segmentation with his corresponding labeling and masking. To achieve it, we used a [Mask R-CNN]([https://arxiv.org/abs/1703.06870](https://arxiv.org/abs/1703.06870)) pre-trained in [COCO Dataset](https://cocodataset.org) (very used for object detection), we fine-tuned and did transfer learning to fit it to our use case, that uses [Cityscapes Dataset](https://www.cityscapes-dataset.com/).

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
- [Future Work](#Future-Work)

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
```python
   labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,      255 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,      255 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,      255 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,      255 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,      255 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,      255 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,      255 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,      255 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,      255 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,      255 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,      255 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,        0 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'rider'                , 25 ,        1 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,        2 , 'vehicle'         , 7       , True         , False        , (255,255,255) ),
    Label(  'truck'                , 27 ,        3 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,        4 , 'vehicle'         , 7       , True         , False        , (  0, 255, 0) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,        5 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,        6 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,        7 , 'vehicle'         , 7       , True         , False        , (  0, 0, 255) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    ]
```

4. Draw the ground truth masks for the training and validation set using:
`cityscapesScripts/cityscapesscripts/preparation/json2labelImg.py`  

![json_dir](https://github.com/MarcKami/aidl-team6-project/blob/master/docs/img/JsonDir.PNG)

```python
mask_list = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"] 
```

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

The decision to use Cityscapes as provider of the Dataset was based on:
  
-   Well-known predictions of the Dataset in other projects and trannings found and well reputation of the provider
-   Big number of images that Dataset contains   
-   Dataset’s folder is ready to use in a project, containing a very good organization   
-   High quality of images   
-   Cityscapes is a dataset of street scenes suitable to develop autonomous driving applications 
-   Most of Cityscapes classes are present in COCO, so the possibility to do Transfer Learning is real    

![](https://lh6.googleusercontent.com/HKka7K_oa49lCw1v_d8WIJHaqcGTB431w9snZ1mev2T99SvXDd7XcDVy9DE4uYK9ffRkbRsdLvHy8NKTlQ25YpwvyRHcsdJ833cjiNzyz55Ohc5MD-Jrk7dQe-sP2NxoazNu9kx1)
 
## Preparation  

It was necessary to apply a layer of Preprocessing in order to prepare the images before using it as Input and Ground Truth in the Neural Network, everything developed in a class ready to get hyper parameters and train or validation image datasets:
	
    class  CityscapesInstanceSegmentation(Dataset)
  

Definitions done in the class are: 
  
-   Label and all meta information about the Dataset  
-   List of all possible labels defined on this project    
-   Mask list  
		
    ```python
    mask_list  = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
    ```
    
-   Take the folder path of each subset of metadata      
    ```python
    for city in os.listdir(self.images_dir):
    	city_images_dir = os.path.join(self.images_dir, city)

    	for image in os.listdir(city_images_dir):
		#example image name: aachen_000000_000019_leftImg8bit.png
   		img_name_prefix = '_'.join(image.split('_')[0:3])

   		#example json file name folder -->/Json_Files/aachen_000168_000019_gtFine_polygons.json
		json_file = os.path.join(self.json_dir, f'{img_name_prefix}_gtFine_polygons.json')

		#example image masks folder -->gtFine/train/aachen_000168_000019_gtFine_polygons
		img_masks_dir = os.path.join(self.masks_dir, f'{img_name_prefix}_gtFine_polygons')              

		self.images.append(os.path.join(city_images_dir, image))              
		self.json_files.append(json_file)
		self.masks.append(img_masks_dir)    
    ```

Then we should use the Masks contained in the Dataset (gtFine folder) to define each target used as outputs in this project:

-   Labels  
    ```python
    def labels_from_mask (self, index):
    ```
    
-   Boxes  
    ```python
    def boxes_from_mask(self, index):
    ```
    
-   Masks
     ```python
    for mask in os.listdir(self.masks[index]):
		mask_path = os.path.join(self.masks[index], mask)
		mask = Image.open(mask_path)
		mask = ImageOps.invert(mask) # 0:background, 1:instance
		masks.append(mask)
    ```


Finally, during the training, the model expects 2 arguments so both are defined in this class:
  
-   Input tensors:  
    Input images converted to RGB and transformed into a Tensor with a defined size
    ```python
    image = Image.open(self.images[index]).convert('RGB')         
    image = transforms.ToTensor()(image)
    ```
  
-   Targets:  
    Output targets are defined and converted into Tensors
    ```python
    masks = [m for m in targets["masks"]]        
    boxes = [b for b in targets["boxes"]]     
    labels = [l for l in targets["labels"]]    

    targets = {"boxes": torch.tensor(np.stack(boxes), dtype= torch.float32, device=device), 
		 "labels": torch.tensor(np.stack(labels), dtype=torch.int64, device=device),
		 "masks": torch.tensor(np.stack(masks), dtype=torch.uint8, device=device)}
    ```


# Model

![](https://github.com/MarcKami/aidl-team6-project/blob/master/docs/img/Heads.png)
	
Our model is a pre-trained Mask-RCNN in COCO Dataset but adapted to our use case. So, to do that, we changed the two last heads of this model to fit it with Cityscapes Dataset.
The implementation has been change the num of classes of the box predictor head and also have the possibility to change the hidden layers and the random range size of the input images.

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

```python
dataset_train = CityscapesInstanceSegmentation(root=HyperParams.dataset_root, split='train')
dataset_val = CityscapesInstanceSegmentation(root=HyperParams.dataset_root, split='val')
```
  
In order to train faster, images have been loaded in memory and then loaded train and validation dataset:

```python
# Load images in memory to train faster
for idx in range(HyperParams.num_samples_train):
  dataset_train.append_images_targets(idx)
for idx in range(HyperParams.num_samples_val):
  dataset_val.append_images_targets(idx)

# Data Loader  
data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=HyperParams.batch_size_train, shuffle='True', num_workers=0,collate_fn=collate_fn)
data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=HyperParams.batch_size_val, shuffle='True', num_workers=0,collate_fn=collate_fn)
```

### Modified Mask-RCNN 

Next step is the model definition, based on a Mask-RCNN: 
```python
model = Team6_MaskRCNN(HyperParams.num_classes, HyperParams.hidden_layer, 
                        HyperParams.min_size, HyperParams.max_size)
```

In the definition of the Model, a pretrained Mask-RCNN model has been used based on Resnet50:
```python
model = detection.maskrcnn_resnet50_fpn(pretrained=pretrained, min_size=min_size, max_size=max_size)
```

Afterwards, next components of the pretrained Mask-RCNN have been replaced:

-   Box Predictor Head
    
-   Mask Predictor Head

```python    
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

### Parameters and Optimizer

Parameters label are defined using the new Fast-CRNN Head Mask and Box predictor and Adam Optimizer has been used in this model: 

```python
parameters = list(model.roi_heads.box_predictor.parameters()) + list(model.roi_heads.mask_predictor.parameters())
optimizer = optim.Adam(parameters, lr=HyperParams.lr, weight_decay=HyperParams.weight_decay)
```
### Train

The train process has as a result a dict of losses where we can see how our model is going through epochs in each kind of task (label, mask, bounding box). So to have a unique loss we sum all the losses present in the dict and then have the big picture of how is it going

```python
for i, (images, targets) in enumerate(dataloader_train):
    optimizer.zero_grad()
    loss_dict = model(images, targets)
    loss = sum(loss for loss in loss_dict.values())            
    scheduler.step(loss)
    loss.backward()
    optimizer.step()
```

In the other hand, to validate, we continue with the model on training mode, bacause on evaluation only returns a prediction without loss, and also only takes as argument an input image. So to solve that, we decided to load the validation data in train mode but without updating the loss neither the optimizer or scheduler.

```python
for i, (images, targets) in enumerate(dataloader_val):
    loss_dict = model(images, targets)
    loss = sum(loss for loss in loss_dict.values())
```

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
We expect quality predictions because the Cityscapes dataset exhibits a high overlap with the COCO dataset. In particular, from the eight classes we want to predict in Cityscapes, seven of them are also present in COCO (person, car, bicycle, motorcycle, truck, bus and train) while only one (rider) is missing. [COCO labels](https://gist.github.com/iitzco/3b2ee634a12f154be6e840308abfcab5)

### Setup
We prepared a set of 6 photos to check the visual results of the inference using this model.
### Results
The pretrained MaskRCNN was able to accurately detect and segmentate the single instances of the above mentioned common classes.  As expected, instances of  `rider` are detected but wrongly classified as "person", which is the most similar class present in COCO
![Result_00](https://github.com/MarcKami/aidl-team6-project/blob/master/docs/exps/Experiment%201/Result_00.PNG)
![Result_06](https://github.com/MarcKami/aidl-team6-project/blob/master/docs/exps/Experiment%201/Result_06.PNG)

### Conclusions
The results confirm our intuition that the model without fine-tuning constitutes a very solid baseline model for the classes present in both datasets.

To make predictions about the class “rider”, which is only found in Cityscapes, we proceed to update the detection and segmentation heads of the MaskRCNN by randomly initializing the last layers and reducing to number of classes to 8 instance classes plus one additional class for the background.  

## Experiment 2
1000 Samples over 20  epochs with (800, 1024) range random size & lr= 0.001 (default Adam)
### Hypothesis
The updated model should recognise the new class “rider” and distinguish it from "person" in the validation images.  
### Setup
We prepare a set of 1000 training images and 200 validation images. As a data augmentation technique, we use random image resizing from 800 and 1024 pixels (short side) as in the MaskRCNN Paper (https://arxiv.org/abs/1703.06870). As optimizer we use Adam with a constant learning rate of 0.001.
### Results
After 4-5 epochs the learning curves stagnate for both training and validation sets, suggesting underfitting.  The fine-tuned model correctly identifies the class "rider" but on the other hand it seems not to recognise class "person". 

- Total Loss:

![](https://github.com/MarcKami/aidl-team6-project/blob/master/docs/exps/Experiment%202/TotalLoss.PNG)
- Splited Losses:

![](https://github.com/MarcKami/aidl-team6-project/blob/master/docs/exps/Experiment%202/SplitLoss.PNG)
- Visual Results:

![](https://github.com/MarcKami/aidl-team6-project/blob/master/docs/exps/Experiment%202/Result_00.PNG)
![](https://github.com/MarcKami/aidl-team6-project/blob/master/docs/exps/Experiment%202/Result_06.PNG)

### Conclusions
To observe further progression a learning rate decay policy may be necessary. 


## Experiment 3
1000 Samples over 10 epochs with (800, 1024) range random size & lr= 0.001 **using ReduceLROnPlateau scheduler**
### Hypotesis
Learning rate decay is a technique that aims the model to converge and avoid oscillations which prevent the optimizer to get to a local minimum By using a Learning Rate Scheduler we expect to observe an improvement in the learning curves, specially in the epochs, where the learning rate is reduced, thus allowing the model to succesfully recognise both "rider" and "person".

### Setup 
We use the same set of 1000 training images and 200 validation images. Starting with the learning rate of 0.001 (as in the previous experiment) we define a Scheduler that reduces it when no significant loss reduction is observed for a few epochs [ReduceLROnPlateau](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html). 
### Results
The behaviour of the learning curves does not match our expectations, as we observe again a stagnation after few epochs of training. Furthermore the model still does not detect the class "person". 

- Total Loss:

![](https://github.com/MarcKami/aidl-team6-project/blob/master/docs/exps/Experiment%203/TotalLoss.PNG)
- Splited Losses:

![](https://github.com/MarcKami/aidl-team6-project/blob/master/docs/exps/Experiment%203/SplitLoss.PNG)
- Visual Results:

![](https://github.com/MarcKami/aidl-team6-project/blob/master/docs/exps/Experiment%203/Result_00.PNG)
![](https://github.com/MarcKami/aidl-team6-project/blob/master/docs/exps/Experiment%203/Result_06.PNG)
### Conclusions
The fact that our model excels with "rider" and does not detect "person at all does not seem plausible: while "person" is one of the most common classes in COCO and is the most frequent class in Cityscapes,  "rider" is ten times less frequent than "person" in Cityscapes and does not appear in COCO at all.  

After checking our complete pipeline we identified the root-cause of this issue: the torchvision MaskRCNN model expects the trainID 0 for the background and we wrongly assigned it to the class "person".

# Final Conclusions
- The out-of-the-shelf pretrained MaskRCNN yields quality predictions on the Cityscapes classes also present in COCO  without any additional training.

- Through fine-tuning it is possible  to successfully detect and segmentate single instances of the non-COCO class “rider”.

- Since the class “person” was wrongly labeled with the ID=0, usually reserved for the background, ROIs containing this class are not considered by the segmentation branch.

# Future Work
- Assign  a valid trainID to the class “person” and repeat the experiments. We expect a better behaviour of the learning curves as “person” is handled as non-background class by the model

- Implement the computation of the metrics “mean average precision” and “average precision per class” to assess the model performance and compare it to the state-of-the-art implementations

- Predict additional non-COCO classes present in the  Cityscapes dataset such as caravan and trailer
