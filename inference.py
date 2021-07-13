import sys
import torch
import os
from PIL import Image
from torchvision.transforms import transforms
from utils import HyperParams, ShowResults, load_model
from model import Team6_MaskRCNN    

if __name__ == "__main__":
    file = str(sys.argv[1])
    #file = str(r'C:\DeepLearning\aidl-team6-project\test-imgs\City\01.jpg')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = Image.open(file).convert('RGB')         
    img = transforms.ToTensor()(img)
    img = torch.tensor(img).to(device)

    # Model
    model = Team6_MaskRCNN(HyperParams.num_classes, HyperParams.hidden_layer, 
                            HyperParams.min_size, HyperParams.max_size)

    # Load Pre-Trained Model
    load_model(model)

    # Show visual results
    ShowResults(model, image=img)



