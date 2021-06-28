import torch
from torch.utils.data import DataLoader

from dataset import MyDataset
from model import MyModel
from utils import accuracy
#from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, get_detection_dataset_dicts
from detectron2.structures import BoxMode
from detectron2.data.datasets import register_coco_instances, register_coco_panoptic, cityscapes_panoptic

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_single_epoch(model, train_loader, optimizer):
    model.train()
    accs, losses = [], []
    for x, y in train_loader:
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        y_ = model(x)
        loss = F.cross_entropy(y_, y)
        loss.backward()
        optimizer.step()
        acc = accuracy(y, y_)
        losses.append(loss.item())
        accs.append(acc.item())
    return np.mean(losses), np.mean(accs)


def eval_single_epoch(model, val_loader):
    accs, losses = [], []
    with torch.no_grad():
        model.eval()
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            y_ = model(x)
            loss = F.cross_entropy(y_, y)
            acc = accuracy(y, y_)
            losses.append(loss.item())
            accs.append(acc.item())
    return np.mean(losses), np.mean(accs)


def train_model(config):
    # Register our CityScapes Datasets
    register_coco_panoptic("my_dataset_train", {}, "C:/DeepLearning/cityscapes/leftImg8bit/train", "C:/DeepLearning/cityscapes/gtFine/cityscapes_panoptic_train", "C:/DeepLearning/cityscapes/gtFine/cityscapes_panoptic_train.json")
    register_coco_panoptic("my_dataset_val", {}, "C:/DeepLearning/cityscapes/leftImg8bit/val", "C:/DeepLearning/cityscapes/gtFine/cityscapes_panoptic_val", "C:/DeepLearning/cityscapes/gtFine/cityscapes_panoptic_val.json")
    register_coco_panoptic("my_dataset_test", {}, "C:/DeepLearning/cityscapes/leftImg8bit/test", "C:/DeepLearning/cityscapes/gtFine/cityscapes_panoptic_test", "C:/DeepLearning/cityscapes/gtFine/cityscapes_panoptic_test.json")

    # Get the dictionary gatasets   
    train_dataset = DatasetCatalog.get("my_dataset_train")
    val_dataset = DatasetCatalog.get("my_dataset_val")
    test_dataset = DatasetCatalog.get("my_dataset_test")

    for d in random.sample(train_dataset, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=None, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2_imshow(out.get_image()[:, :, ::-1])

    #data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    #my_dataset = MyDataset("session-2/data/data/data/", "session-2/data/chinese_mnist.csv", transform=data_transforms)
    #train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(my_dataset, [10000, 2500, 2500])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    my_model = MyModel(config["h1"], config["h2"], config["h3"], config["h4"]).to(device)

    optimizer = optim.Adam(my_model.parameters(), config["lr"])
    for epoch in range(config["epochs"]):
        loss, acc = train_single_epoch(my_model, train_loader, optimizer)
        print(f"Train Epoch {epoch} loss={loss:.2f} acc={acc:.2f}")
        loss, acc = eval_single_epoch(my_model, val_loader)
        print(f"Eval Epoch {epoch} loss={loss:.2f} acc={acc:.2f}")
    
    loss, acc = eval_single_epoch(my_model, test_loader)
    print(f"Test loss={loss:.2f} acc={acc:.2f}")

    return my_model


if __name__ == "__main__":

    config = {
        "lr": 1e-3,
        "batch_size": 64,
        "epochs": 5,
        "h1": 32,
        "h2": 64,
        "h3": 128,
        "h4": 128,
    }
    my_model = train_model(config)

    
print("Hola Mundo")