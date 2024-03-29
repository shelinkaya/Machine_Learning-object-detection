##importing of the required packages
from pycocotools.coco import COCO
import numpy as np 
import pandas as pd 
import os
import torch
import torchvision
import sys
from torchvision import datasets, models
from torchvision.transforms import functional as FT
from torchvision.transforms import transforms as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split, Dataset
import copy
import math
import cv2
from PIL import Image
import albumentations as A
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict, deque
import datetime
import time
from tqdm import tqdm
from torchvision.utils import draw_bounding_boxes
from albumentations.pytorch import ToTensorV2

#collate the input samples into a batch
def collate_fn(batch):
    return tuple(zip(*batch))
##transformation of size,color, brightness etc.

def get_albumentation(train):
    if train:
        transform = A.Compose([
            A.Resize(500, 500),
            A.ColorJitter(p=0.1),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    else:
        transform = A.Compose([
            A.Resize(500, 500),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    return transform
##defining of class of fiber data

class FiberDetection(datasets.VisionDataset):
    def __init__(
        self,
        root: str,
        split = "train",
        transform= None,
        target_transform = None,
        transforms = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.split = split
        self.coco = COCO(os.path.join(root, split, "annotations.json"))
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.ids = [id for id in self.ids if (len(self._load_target(id)) > 0)]

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        image = cv2.imread(os.path.join(self.root, self.split, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _load_target(self, id: int):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int):
        id = self.ids[index]
        image = self._load_image(id)
        target = copy.deepcopy(self._load_target(id))
        
        boxes = [t['bbox'] + [t['category_id']] for t in target]
        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes)
        
        image = transformed['image']
        boxes = transformed['bboxes']
        new_boxes = []
        for box in boxes:
            xmin =  box[0]
            ymin = box[1]
            xmax = xmin + box[2]
            ymax = ymin + box[3]
            new_boxes.append([xmin, ymin, xmax, ymax])
        
        boxes = torch.tensor(new_boxes, dtype=torch.float32)
        
        targ = {}
        targ["boxes"] = boxes
        targ["labels"] = torch.tensor([t["category_id"]  for t in target], dtype=torch.int64)
        targ["image_id"] = torch.tensor([t["image_id"]  for t in target])
        targ["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        targ["iscrowd"] = torch.tensor([t["iscrowd"]  for t in target], dtype=torch.int64)

        return image.div(255), targ


    def __len__(self) -> int:
        return len(self.ids)
        
#define dataset path here and assign cocos to .json file

dataset_path = "fiber_dataset"
coc = COCO(os.path.join(dataset_path,"train","annotations.json"))

##printing of categories and classes
categories = coc.cats
n_classes = len(categories.keys())
n_classes, categories

##add name to given list
classes = []
for i in categories.items():
    classes.append(i[1]["name"])
    
##define number of classes and categories
n_classes= 2
categories=1

##define your train, validation and test dataset
train_dataset = FiberDetection(root=dataset_path,split="train", transforms=get_albumentation(True))
val_dataset = FiberDetection(root=dataset_path, split="valid", transforms=get_albumentation(False))
test_dataset = FiberDetection(root=dataset_path, split="test", transforms=get_albumentation(False))

#plot an image from your train_dataset it will be our ground truth

sample = train_dataset[15]
real = torch.tensor(sample[0] * 255, dtype=torch.uint8)

fig = plt.figure(figsize=(10, 10))
plt.imshow(draw_bounding_boxes(real, 
                               sample[1]['boxes'],
                               width=4).permute(1, 2, 0)
          )
          
## apply fasterrcnn here with pretrained model

model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes) 

##define your loader files with number of workers, collete fn etc.

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1,shuffle=True, num_workers=0,
    collate_fn=collate_fn
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1,shuffle=True, num_workers=0,
    collate_fn=collate_fn
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1,shuffle=True, num_workers=0,
    collate_fn=collate_fn
)

##assign the images and targets to model
dataiter = iter(train_loader)
images, targets = dataiter.next()


images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]

output = model(images,targets)

#evaluation of the model
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
model.eval()
predictions = model(x)

#adjustment of device gpu. use it if cpu is available

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device

#assign model to the device
model = model.to(device)

##define parameteres, optmizers, and learning rate schedular
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001,
                            momentum=0.9, nesterov=True, weight_decay=1e-4)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16, 22], gamma=0.1)

#train your model

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.to(device)
    model.train()
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    all_losses = []
    all_losses_dict = []
    
    for images, targets in tqdm(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
        losses_reduced = sum(loss for loss in loss_dict.values())

        loss_value = losses_reduced.item()
        
        all_losses.append(loss_value)
        all_losses_dict.append(loss_dict_append)
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
    
    all_losses_dict = pd.DataFrame(all_losses_dict)
    print("Epoch {}: lr: {:.6f} loss: {:.6f}, loss_classifier: {:.6f}, loss_box_reg: {:.6f}, loss_rpn_box_reg: {:6f}, loss_objectness: {:.6f}".format(
        epoch, optimizer.param_groups[0]["lr"], np.mean(all_losses), 
        all_losses_dict["loss_classifier"].mean(),
        all_losses_dict["loss_box_reg"].mean(),
        all_losses_dict["loss_rpn_box_reg"].mean(),
        all_losses_dict["loss_objectness"].mean(),
    ))
    
    
 #define number of epoch and run your model training
num_epochs = 5

for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=20)
    lr_scheduler.step()
    
#evaluation of  model
model.eval()
torch.cuda.empty_cache()

##apply trained model on your test dataset

img, _ = test_dataset[15]
img_int = torch.tensor(img * 255, dtype=torch.uint8)
with torch.no_grad():
    prediction = model([img.to(device)])
    pred = prediction[0]
    
#plot image here
fig = plt.figure(figsize=(10,10))
plt.imshow(draw_bounding_boxes(img_int, 
                               pred['boxes'],
                               width=3).permute(1, 2, 0)
          )
          
