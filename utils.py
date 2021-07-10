import torch
import numpy as np
import torch
import random
import torchvision
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from dataset import CityscapesInstanceSegmentation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HyperParams():
    dataset_root = "C:\DeepLearning\cityscapes"
    model_folder = r"C:\DeepLearning\aidl-team6-project\models"
    model_good = r"C:\DeepLearning\aidl-team6-project\models\model_good"
    num_samples = 4
    batch_size_train = 2
    batch_size_val = 2
    num_classes = 1+8
    hidden_layer = 256
    min_size = 400
    max_size = 600
    lr = 0.0003
    weight_decay = 0.0001
    epochs = 1

def save_model(model, optimizer, loss, path):
    checkpoint = { 
        "model_state_dict": model.state_dict(), 
        "optimizer_state_dict": optimizer.state_dict(), 
        "loss": loss}
    torch.save(checkpoint, path)

def load_model(model, optimizer, loss = 0):
    checkpoint = torch.load(HyperParams.model_good, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def collate_fn(batch):
    images = []
    targets = []
    for i, t in batch:
        images.append(i)
        targets.append(t)
    return images, targets

def ShowResults(model, sample):
    with torch.no_grad():
        model = model.cpu()
        model.eval()
        img,targets = sample
        detections = model([img.cpu()])

    detections= detections[0]
        
    iou_threshold = 0.2 #0.2
    scores_threshold = 0.8

    keep_idx = torchvision.ops.nms(detections['boxes'], detections['scores'], iou_threshold)
    boxes = [b for i,b in enumerate(detections['boxes']) if i in keep_idx]
    labels = [l for i,l in enumerate(detections['labels']) if i in keep_idx]
    masks = [m for i,m in enumerate(detections['masks']) if i in keep_idx]

    img = to_pil_image(img)
    transformed_img_copy = img.copy()

    label2name={label.trainId:label.name for label in CityscapesInstanceSegmentation.labels if label.name in CityscapesInstanceSegmentation.mask_list}
    label2color={label.trainId:label.color for label in CityscapesInstanceSegmentation.labels if label.name in CityscapesInstanceSegmentation.mask_list}

    #overlap masks
    for box, label, mask in zip(boxes, labels, masks):
        mask = (mask.cpu().numpy()*50).astype(np.int8)  
        mask = mask.squeeze(0)
        mask_im = Image.fromarray(mask, mode="L")
        full_color = Image.new("RGB", transformed_img_copy.size, (0, 255, 0))
        transformed_img_copy = Image.composite(full_color, transformed_img_copy, mask_im)
    
    plt.figure(figsize=(20,10))  
    plt.imshow(np.asarray(transformed_img_copy))

    print(label2name)
    print(len(CityscapesInstanceSegmentation.mask_list))
    labels = [l.item() for l in labels]
    print(f"ground truth labels: {targets['labels']}, len: {len(targets['labels'])}")
    print(f'predicted labels: {labels}, len: {len(labels)}')

    COLORS = np.random.uniform(128, 255, size=(100, 3))
    alpha = 1 
    beta = 0.6 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum

    image = img.copy()

    boxes = [[(int(x1),int(y1)),(int(x2),int(y2))] for x1,y1,x2,y2 in boxes]

    for i,label in enumerate(labels):
        if label not in range(len(CityscapesInstanceSegmentation.mask_list)):
            pass
        else:
            label2name={label.trainId:label.name for label in CityscapesInstanceSegmentation.labels}
            label_name = label2name[label]
            masks[i] = masks[i].squeeze(0)
            masks[i] = torch.round(masks[i])
            red_map = np.zeros_like(masks[i]).astype(np.uint8)
            green_map = np.zeros_like(masks[i]).astype(np.uint8)
            blue_map = np.zeros_like(masks[i]).astype(np.uint8)
            # apply a randon color mask to each object
            color = COLORS[random.randrange(0, len(COLORS))]
            masks[i] = np.asarray(masks[i])
            red_map[np.nonzero(masks[i])], green_map[np.nonzero(masks[i])], blue_map[np.nonzero(masks[i])] = color

            # combine all the masks into a single image
            segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
            #convert the original PIL image into NumPy format
            image = np.array(image)
            # convert from RGN to OpenCV BGR format
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # apply mask on the image
            cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)
            # draw the bounding boxes around the objects
            print(color)
            
            cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color, thickness=2)
            # put the label text above the objects

            cv2.putText(image, label_name, (boxes[i][0][0], boxes[i][0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2, cv2.LINE_AA)

    plt.figure(figsize=(20,10))
    plt.imshow(np.asarray(image))