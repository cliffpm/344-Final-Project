import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

csv_path = "/mnt/data/GroundTruth.csv"

df = pd.read_csv(csv_path)


# image_path | mask_path | label

df["label_idx"] = df["label"].astype("category").cat.codes

train_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["label_idx"])
train_df, val_df  = train_test_split(train_df, test_size=0.15, random_state=42, stratify=train_df["label_idx"])

class SkinLesionDataset(Dataset):
    def __init__(self, df, transform_img=None, transform_mask=None):
        self.df = df
        self.transform_img = transform_img
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row["image_path"]).convert("RGB")
        mask = Image.open(row["mask_path"]).convert("L")  # grayscale mask

        if self.transform_img:
            img = self.transform_img(img)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        label = torch.tensor(row["label_idx"], dtype=torch.long)
        return img, mask, label

img_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

mask_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

train_ds = SkinLesionDataset(train_df, img_transform, mask_transform)
val_ds   = SkinLesionDataset(val_df, img_transform, mask_transform)
test_ds  = SkinLesionDataset(test_df, img_transform, mask_transform)

train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl   = DataLoader(val_ds, batch_size=16)
test_dl  = DataLoader(test_ds, batch_size=16)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
       
        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(128, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(128+128, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(64+64, 64)

        self.seg_head = nn.Conv2d(64, 1, 1)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        # Encoder
        c1 = self.down1(x)
    def forward(self, x):
        # Encoder
        c1 = self.down1(x)
        p1 = self.pool1(c1)

        c2 = self.down2(p1)
        p2 = self.pool2(c2)

        bn = self.bottleneck(p2)

        # Classification branch (from bottleneck)
        class_out = self.classifier(bn)

        # Decoder
        u2 = self.up2(bn)
        u2 = torch.cat([u2, c2], dim=1)
        c3 = self.conv2(u2)

        u1 = self.up1(c3)
        u1 = torch.cat([u1, c1], dim=1)
        c4 = self.conv1(u1)

        seg_out = torch.sigmoid(self.seg_head(c4))

        return seg_out, class_out

model = UNet(n_classes=df["label_idx"].nunique())
model = model.cuda()

seg_criterion = nn.BCELoss()
cls_criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    for imgs, masks, labels in train_dl:
        imgs, masks, labels = imgs.cuda(), masks.cuda(), labels.cuda()

        seg_pred, cls_pred = model(imgs)

        loss_seg = seg_criterion(seg_pred, masks)
        loss_cls = cls_criterion(cls_pred, labels)

        loss = loss_seg + loss_cls

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch", epoch, "Loss:", loss.item())

model.eval()
with torch.no_grad():
    img, mask, label = test_ds[0]
    img = img.unsqueeze(0).cuda()
    seg, pred_class = model(img)
    pred_class = pred_class.argmax(dim=1).item()

print("Predicted class:", pred_class)
