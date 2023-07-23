# %%
from typing import LiteralString

import cv2
import os
import json
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import numpy as np
from tqdm import tqdm

# %%
folder_path = '../9517proj_sources/train'
annotation_path = '../9517proj_sources/train_annotations'

with open(annotation_path, 'r') as f:
    annotations = json.load(f)

for anno in annotations:
    image_id = anno['image_id']
    bbox = anno['bbox']
    category_id = anno['category_id']

    # build image path, absolute path+ na me
    image_file_path = os.path.join(folder_path, f"image_id_{str(image_id).zfill(3)}.jpg")

    image = cv2.imread(image_file_path)

    if image is None:
        print(f"File not found: {image_file_path}")
        continue

    # draw bounding box
    x, y, width, height = bbox
    cv2.rectangle(image, (int(x), int(y)), (int(x + width), int(y + height)), (0, 255, 0), 2)

    # show image


# %%
class CustomDataset(Dataset):
    def __init__(self, img_folder, img_ext, mask_folder, mask_ext, transform=None):
        self.img_folder = img_folder
        self.img_ext = img_ext
        self.mask_folder = mask_folder
        self.mask_ext = mask_ext
        self.transform = transform
        self.filenames = [os.path.splitext(filename)[0] for filename in os.listdir(img_folder)]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, self.filenames[idx] + self.img_ext)
        mask_path = os.path.join(self.mask_folder, self.filenames[idx] + self.mask_ext)
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # assuming masks are in 'L' mode

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = torch.squeeze(mask, 0)  # remove the first dimension (1, H, W) -> (H, W)

        return image, mask


transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

path_folder = '../9517proj_sources/train'

path_folder_2 = '../9517proj_sources/off'

train_dataset = CustomDataset(path_folder, '.jpg', path_folder_2, '.png', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# create model
model = deeplabv3_resnet50(pretrained=False, progress=True, num_classes=2, aux_loss=None)
model = model.to('cuda') if torch.cuda.is_available() else model.to('cpu')

# Loss Function and Optimizer
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.0001)

for epoch in range(10):  # suppose we train for 10 epochs
    try:
        for images, masks in train_loader:
            print(images.shape, masks.shape)
            images = images.to('cuda') if torch.cuda.is_available() else images
            masks = masks.to('cuda') if torch.cuda.is_available() else masks

            # forward pass
            outputs = model(images)['out']
            loss = criterion(outputs, masks.long())  # note we need to convert masks to 'long' type

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch [{}/10], Loss: {:.4f}'.format(epoch + 1, loss.item()))
    except Exception as e:
        print(e)
        continue
# %%

# %%
torch.save(model.state_dict(), 'model_weights.pth')
# %%
# model.eval()
test_folder: LiteralString = '../9517proj_sources/valid/valid'
test_ext = '.jpg'
os.makedirs('output_folder', exist_ok=True)
output_folder = '../9517proj_sources/output_folder'

test_filenames = [os.path.splitext(filename)[0] for filename in os.listdir(test_folder)]

with torch.no_grad():  # we don't need gradients for testing
    for filename in test_filenames:
        img_path = os.path.join(test_folder, filename + test_ext)
        image = Image.open(img_path).convert('RGB')
        orig_size = (image.width, image.height)
        image = transform(image)  # apply the same transform as during training
        image = image.unsqueeze(0)  # add a batch dimension

        image = image.to('cuda') if torch.cuda.is_available() else image

        output = model(image)['out']
        output = torch.argmax(output, dim=1)  # get the most likely prediction

        # resize the output to match the original image size
        output = cv2.resize(output[0].cpu().numpy(), orig_size, interpolation=cv2.INTER_NEAREST)
        # print(np.unique(output))
        # print(output)
        # get the original image
        orig_img = cv2.imread(img_path)

        # apply the mask to the original image
        segmented_img = np.zeros_like(orig_img)
        for i in range(3):  # for each color channel
            segmented_img[:, :, i] = np.where(output == 1, orig_img[:, :, i], 0)
        # print(np.unique(segmented_img))
        # save the segmented image
        # segmented_img = (segmented_img * 255).astype(np.uint8)
        res = cv2.imwrite(filename + '_segmented.jpg', segmented_img)
        print(res)
