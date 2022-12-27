import torch
import math
from typing import List

from torch import nn
import numpy as np

from tqdm import tqdm
import wandb

device = torch.device('cuda:0')

from pycocotools.coco import COCO
import torchvision.transforms.functional as TF

train_annotations = COCO("/data/coco2017/annotations/instances_train2017.json")
valid_annotations = COCO("/data/coco2017/annotations/instances_val2017.json")

cat_ids = train_annotations.getCatIds(supNms=["person", "vehicle"])
train_img_ids = []
for cat in cat_ids:
    train_img_ids.extend(train_annotations.getImgIds(catIds=cat))
    
train_img_ids = list(set(train_img_ids))
print(f"Number of training images: {len(train_img_ids)}")

valid_img_ids = []
for cat in cat_ids:
    valid_img_ids.extend(valid_annotations.getImgIds(catIds=cat))
    
valid_img_ids = list(set(valid_img_ids))
print(f"Number of validation images: {len(valid_img_ids)}")


from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple
import cv2
class ImageData(Dataset):
    def __init__(
        self, 
        annotations: COCO, 
        img_ids: List[int], 
        cat_ids: List[int], 
        root_path: Path, 
        transform: Optional[Callable]=None
    ) -> None:
        super().__init__()
        self.annotations = annotations
        self.img_data = annotations.loadImgs(img_ids)
        self.cat_ids = cat_ids
        self.files = [str(root_path / img["file_name"]) for img in self.img_data]
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        ann_ids = self.annotations.getAnnIds(
            imgIds=self.img_data[i]['id'], 
            catIds=self.cat_ids, 
            iscrowd=None
        )
        anns = self.annotations.loadAnns(ann_ids)
        mask = torch.LongTensor(np.max(np.stack([self.annotations.annToMask(ann) * ann["category_id"] 
                                                 for ann in anns]), axis=0)).unsqueeze(0)
        
        #img = io.read_image(self.files[i])
        img = cv2.imread(self.files[i])
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] == 1:
            img = torch.cat([img]*3)
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        #print(img.shape)
        if self.transform is not None:
            return self.transform(img, mask)
        
        return img, mask
def train_transform(
    img1: torch.LongTensor,
    img2: torch.LongTensor
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    
    img1 = TF.resize(img1, size=IMAGE_SIZE)
    img2 = TF.resize(img2, size=IMAGE_SIZE)

    return img1, img2


ROOT_PATH = Path("/data/coco2017/")
BATCH_SIZE = 512
IMAGE_SIZE = (128, 128)

train_data = ImageData(train_annotations, train_img_ids, cat_ids, ROOT_PATH / "train2017", train_transform)
valid_data = ImageData(valid_annotations, valid_img_ids, cat_ids, ROOT_PATH / "val2017", train_transform)

train_dl = DataLoader(
    train_data,
    BATCH_SIZE, 
    shuffle=True, 
    drop_last=True, 
    num_workers=4,
    pin_memory=True,
)

valid_dl = DataLoader(
    valid_data,
    BATCH_SIZE, 
    shuffle=False, 
    drop_last=False, 
    num_workers=4,
    pin_memory=True,
)


model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)

model.to(device=device)

# Set device: `cuda` or `cpu`
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = torch.nn.BCEWithLogitsLoss()

#optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.00008),])


import ranger21
optimizer = ranger21.Ranger21(
            model.parameters(),
            lr=1e-3,
            num_epochs=100,
            num_batches_per_epoch=579,
            use_madgrad=False)

class engine():
    def train_batch(model, data, optimizer, criterion, graphed_model=None):
        ims, ce_masks = data
        ims = ims.to(device=device, dtype=torch.float32)
        ce_masks = ce_masks.to(device=device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        if graphed_model is None:
            _masks = model(ims)
            optimizer.zero_grad()
            loss = criterion(_masks, ce_masks.float())
            loss.backward()
            optimizer.step()
        else:
            _masks = graphed_model(ims)
            loss = criterion(_masks, ce_masks.float())
            loss.backward()
            optimizer.step()
        return loss.item()


    @torch.no_grad()
    def validate_batch(model, data, criterion):
        ims, masks = data
        ims = ims.to(device=device, dtype=torch.float32)
        masks = masks.to(device=device, dtype=torch.long)
        _masks = model(ims)
        loss = criterion(_masks, masks.float())
        return loss.item()

name = "without_cuda_graph_adam"
#name = "with_cuda_graph_ranger21"

# Set num of epochs
EPOCHS = 100
# remove if you don't use wandb
wandb.init(project="cuda_graph", entity="", name=name)

def run():
    model.train()
    dummy_input = torch.rand(BATCH_SIZE, 3, IMAGE_SIZE[0], IMAGE_SIZE[0]).to(device=device)
    graphed_model = torch.cuda.make_graphed_callables(model, (dummy_input,))
    #graphed_model=None

    for epoch in range(EPOCHS):
        print("####################")
        print(f"       Epoch: {epoch}   ")
        print("####################")

        print(len(train_dl))
        train_losses = []
        val_losses = []

        model.train()

        for bx, data in tqdm(enumerate(train_dl), total = len(train_dl)):
            train_loss = engine.train_batch(model, data, optimizer, criterion, graphed_model=graphed_model)
            train_losses.append(train_loss)

        model.eval()
        for bx, data in tqdm(enumerate(valid_dl), total = len(valid_dl)):
            val_loss = engine.validate_batch(model, data, criterion)
            val_losses.append(val_loss)

        # remove if you don't use wandb
        wandb.log({ 
              'epoch': epoch,
              'train_loss': sum(train_losses) / len(train_losses),
              'val_loss': sum(val_losses) / len(val_losses)})

run()