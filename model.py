import torch
import os
import torch.nn as nn
import torchvision.models.detection as detection
from collections import OrderedDict
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils import HyperParams, Losses, plot_losses, save_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Team6_MaskRCNN(num_classes: int, hidden_layer: int, min_size: int, max_size: int, pretrained = True):
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

    return model

def train_model(model, optimizer, dataloader_train, dataloader_val):
    model.to(device)
    model.train()
    train_losses = Losses()
    val_losses = Losses()
    for e in range(HyperParams.epochs):
        # TRAIN
        print('Training epoch ' + str(e))
        train_losses.reset()
        for i, (images, targets) in enumerate(dataloader_train):
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            optimizer.step()
            
            if i%1 == 0:
                loss_dict_printable = {k: f"{v.item():.2f}" for k, v in loss_dict.items()}
                print(f"[{(i+1)*dataloader_train.batch_size}/{len(dataloader_train.dataset)}] loss: {loss_dict_printable}")

            train_losses.sum(loss, loss_dict, len(images))

        train_losses.mean(len(dataloader_train.dataset))
        train_losses.toList()

        # EVAL
        print('Evaluating epoch ' + str(e))        
        val_losses.reset()
        for i, (images, targets) in enumerate(dataloader_val):
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            
            if i%1 == 0:
                loss_dict_printable = {k: f"{v.item():.2f}" for k, v in loss_dict.items()}
                print(f"[{(i+1)*dataloader_val.batch_size}/{len(dataloader_val.dataset)}] loss: {loss_dict_printable}")   
            
            val_losses.sum(loss, loss_dict, len(images))

        val_losses.mean(len(dataloader_val.dataset))
        val_losses.toList()

        # # Save model
        # name = 'model_epoch_' + str(e+1)
        # save_model(model, optimizer, train_losses, val_losses, path=os.path.join(HyperParams.model_folder, name))
        # print('Model ' + name + ' saved!')
    plot_losses(train_losses, val_losses)