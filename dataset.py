import torch
import numpy as np
import json
import os
from collections import namedtuple
from typing import Any, Callable, Dict, Optional
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import transforms

#from cityscapesScripts.cityscapesscripts.preparation.labels import Label
#from cityscapesScripts.cityscapesscripts.preparation.json2labelImg import mask_list  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CityscapesInstanceSegmentation(Dataset):
    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if mode="fine"
            otherwise ``train``, ``train_extra`` or ``val``
        mode (string, optional): The quality mode to use, ``fine`` or ``coarse``
        target_type (string or list, optional): Type of target to use, ``instance``, ``semantic``, ``polygon``
            or ``color``. Can also be a list to output a tuple with all specified target types.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    Examples:

        Get semantic segmentation target

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type='semantic')

            img, smnt = dataset[0]

        Get multiple targets

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type=['instance', 'color', 'polygon'])

            img, (inst, col, poly) = dataset[0]

        Validate on the "coarse" set

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='val', mode='coarse',
                                 target_type='semantic')

            img, smnt = dataset[0]
    """



# a label and all meta information
    Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

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

    mask_list = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"] 

    def __init__(
            self,
            root: str,
            split: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ):
    #NEW
        super(CityscapesInstanceSegmentation, self).__init__()
        
        self.root = root
        self.split = split
        self.images_dir = os.path.join(self.root, 'leftImg8bit', self.split)
        #print(f'images_dir={self.images_dir}')
        self.json_dir = os.path.join(self.root, 'Json_Files', self.split)
        #print(f'json_dir={self.json_dir}')
        self.masks_dir = os.path.join(self.root, 'gtFine', self.split)
        #print(f'masks_dir={self.masks_dir}')   
        
        self.images = []
        self.json_files = []
        self.masks = []

        self.img = []
        self.targets = []

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.masks_dir): # or not os.path.isdir(self.json_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                                   ' specified "split" are inside the "root" directory')
        else: pass

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


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        
           During training, the model expects both the input tensors, as well as a targets (list of dictionary),
          containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance
        
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """  

        """
        #print(f'Image path{self.images[index]}')        
        image = Image.open(self.images[index]).convert('RGB')         
        #convert to tensor in the range [0.0, 1.0]
        image = transforms.ToTensor()(image).to(device)  #torch.Size([3,1024, 2048])
        
        #targets
        labels = self.labels_from_mask(index)
        boxes = self.boxes_from_mask(index) 
        masks=[]
        for mask in os.listdir(self.masks[index]):
          mask_path = os.path.join(self.masks[index], mask)
          mask = Image.open(mask_path)
          mask = ImageOps.invert(mask) # 0:background, 1:instance
          masks.append(mask)

        targets = {"boxes": torch.tensor(boxes, dtype= torch.float32, device=device), 
                   "labels": torch.tensor(labels, dtype=torch.int64, device=device),
                   "masks": torch.tensor(np.stack(masks), dtype=torch.uint8, device=device)}
        
        masks = [m for m in targets["masks"]] 
        norm_masks = []
        for mask in masks:  
          mask = np.asarray(mask.cpu())     
          max = np.max(mask)
          min = np.min(mask)
          mask = np.array([(x - min) / (max - min) for x in mask])
          norm_masks.append(mask)

        targets["masks"] = torch.tensor(np.stack(norm_masks), dtype=torch.uint8, device=device)
        """

        image = self.img[index]
        targets = self.targets[index]

        image = torch.tensor(image).to(device)

        masks = [m for m in targets["masks"]]        
        boxes = [b for b in targets["boxes"]]     
        labels = [l for l in targets["labels"]]    

        targets = {"boxes": torch.tensor(np.stack(boxes), dtype= torch.float32, device=device), 
                  "labels": torch.tensor(np.stack(labels), dtype=torch.int64, device=device),
                  "masks": torch.tensor(np.stack(masks), dtype=torch.uint8, device=device)}

        return image, targets

    def append_images_targets (self, index: int):
        #print(f'Image path{self.images[index]}')        
        image = Image.open(self.images[index]).convert('RGB')         
        #convert to tensor in the range [0.0, 1.0]
        #image = transforms.ToTensor()(image).to(device)  #torch.Size([3,1024, 2048])
        image = transforms.ToTensor()(image)  #torch.Size([3,1024, 2048])
        
        #targets
        labels = self.labels_from_mask(index)
        boxes = self.boxes_from_mask(index) 

        masks=[]
        for mask in os.listdir(self.masks[index]):
          mask_path = os.path.join(self.masks[index], mask)
          mask = Image.open(mask_path)
          mask = ImageOps.invert(mask) # 0:background, 1:instance
          masks.append(mask)

        # targets = {"boxes": torch.tensor(boxes, dtype= torch.float32, device=device), 
        #            "labels": torch.tensor(labels, dtype=torch.int64, device=device),
        #            "masks": torch.tensor(np.stack(masks), dtype=torch.uint8, device=device)}

        targets = {"boxes": torch.tensor(boxes, dtype= torch.float32, device='cpu'), 
                   "labels": torch.tensor(labels, dtype=torch.int64, device='cpu'),
                   "masks": torch.tensor(np.stack(masks), dtype=torch.uint8, device='cpu')}
        
        masks = [m for m in targets["masks"]] 
        norm_masks = []
        for mask in masks:  
          #mask = np.asarray(mask.cpu())  
          mask = np.asarray(mask)     
          max = np.max(mask)
          min = np.min(mask)
          mask = np.array([(x - min) / (max - min) for x in mask])
          norm_masks.append(mask)

        #targets["masks"] = torch.tensor(np.stack(norm_masks), dtype=torch.uint8, device=device)
        targets["masks"] = torch.tensor(np.stack(norm_masks), dtype=torch.uint8, device='cpu')

        self.img.append(image)
        self.targets.append(targets)
        print("Appended image and target " + str(index))

        return image, targets

    def labels_from_mask (self, index):
      labels = []

      for mask in os.listdir(self.masks[index]):
        label = int(mask.split('_')[0])
        labels.append(label)
      return labels


    def labels_from_json (self, index):
      data = self._load_json(self.json_files[index])
      #data = {'imgHeight': 1024, 'imgWidth': 2048, 'objects': [{'label': 'building', polygon': [[0, 0], [2047, 0], [2047, 693], [0, 666]]}, {'label': 'road'

      # Create dictionaries for a fast lookup
      name2label={label.name:label for label in CityscapesInstanceSegmentation.labels}

      labels = []

      for item in data['objects']:
        if not item['label'] in CityscapesInstanceSegmentation.mask_list:
          pass
        else:
          name = item['label']
          label= name2label[name].trainId 
          labels.append(label)
      return labels


    def boxes_from_mask(self, index):
      boxes = []
      for mask in os.listdir(self.masks[index]):
        mask_path = os.path.join(self.masks[index], mask)
        mask = Image.open(mask_path)
        mask = ImageOps.invert(mask) 
        mask = np.asarray(mask)
        pos = np.nonzero(mask)
        x1 = np.min(pos[1])
        x2 = np.max(pos[1])
        y1 = np.min(pos[0])
        y2 = np.max(pos[0])
        box = [x1,y1,x2,y2]
        boxes.append(box)
      return boxes

    def boxes_from_json(self, index):
      data = self._load_json(self.json_files[index])
      #data = {'imgHeight': 1024, 'imgWidth': 2048, 'objects': [{'label': 'building', polygon': [[0, 0], [2047, 0], [2047, 693], [0, 666]]}, {'label': 'road', 'polygon':
      
      boxes = []
      #boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with 0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``
      
      for item in data['objects']:  
        if not item['label'] in CityscapesInstanceSegmentation.mask_list: pass
          #print(f"label: {item['label']}")
      
        else:
          polygon = item['polygon']
          #print(f'polygon {polygon}')
          x1 = np.minimum.reduce(polygon)[0]
          x2 = np.maximum.reduce(polygon)[0]
          y1 = np.minimum.reduce(polygon)[1]
          y2 = np.maximum.reduce(polygon)[1]
          box = [x1,y1,x2,y2]
          #print(f"label: {item['label']}, box: {box}")
          boxes.append(box)
      #print(f'Num boxes: {len(boxes)}')
      return boxes

    def __len__(self) -> int:
        #return len(self.images)
        return len(self.img)

    def extra_repr(self) -> str:
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return '\n'.join(lines).format(**self.__dict__)

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as file:
            data = json.load(file)
        return data

