#%%
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

#%%
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
    cv2.rectangle(image, (int(x), int(y)), (int(x+width), int(y+height)), (0, 255, 0), 2)

    # show image
#%%
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
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

path_folder = '../9517proj_sources/train'

path_folder_2 = '../9517proj_sources/off'

train_dataset = CustomDataset(path_folder,'.jpg',path_folder_2, '.png',transform=transform)
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
#%%

#%%
torch.save(model.state_dict(), 'model_weights.pth')
#%%
test_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
model.eval()
output_folder = []
os.makedirs('output', exist_ok=True)

# 检查是否有可用的 GPU
if torch.cuda.is_available():
    device = torch.device('cuda')  # 使用 GPU
else:
    device = torch.device('cpu')   # 使用 CPU


with torch.no_grad():
    for i, (images) in enumerate(test_loader):  # 如果测试集没有标签，则不需要 '_'
        # images_tensor = torch.stack([torch.from_numpy(i) for i in images])
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs.data, 1)  # 获取预测的最大概率类别

        # 将结果转换回 PIL 图像
        img = transforms.ToPILImage()(predicted.squeeze().cpu())

        # 保存图像
        img.save(os.path.join(output_folder, f'image_{i}.png'))
