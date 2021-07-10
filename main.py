import torch
from dataset import CityscapesInstanceSegmentation
from utils import HyperParams, collate_fn, ShowResults, load_model
from model import Team6_MaskRCNN, train_model
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Dataset
dataset_train = CityscapesInstanceSegmentation(root=HyperParams.dataset_root, split='train')
dataset_val = CityscapesInstanceSegmentation(root=HyperParams.dataset_root, split='val')

# Load images in memory to train faster
for idx in range(HyperParams.num_samples):
  dataset_train.append_images_targets(idx)
  dataset_val.append_images_targets(idx)

# Data Loader  
data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=HyperParams.batch_size_train, shuffle='True', num_workers=0,collate_fn=collate_fn)
data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=HyperParams.batch_size_val, shuffle='True', num_workers=0,collate_fn=collate_fn)

# Model
model = Team6_MaskRCNN(HyperParams.num_classes, HyperParams.hidden_layer, 
                        HyperParams.min_size, HyperParams.max_size)

# Optimizer
parameters = list(model.roi_heads.box_predictor.parameters()) + list(model.roi_heads.mask_predictor.parameters())
optimizer = optim.Adam(parameters, lr=HyperParams.lr, weight_decay=HyperParams.weight_decay)

# Train
train_model(model, optimizer, data_loader_train, data_loader_val)

# Load Pre-Trained Model
#load_model(model, optimizer)

# Show visual results
ShowResults(model, dataset_val[3])