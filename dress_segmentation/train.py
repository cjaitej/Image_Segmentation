import torch
import os
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from UNet import UNet
from utils import (
    save_checkpoint,
    get_loader,
)

#HyperParameters
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 64
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "C:/Data/one_more_hd/train/cloth"
TRAIN_MASK_DIR = "C:/Data/one_more_hd/train/cloth-mask"

def train(loader, model, optimizer, loss_fn):
    loop = tqdm(loader)
    running_loss = 0
    for batch, (data, target) in enumerate(loop):
        data = data.to(DEVICE, dtype=torch.float)
        target = target.float().to(DEVICE)

        with torch.cuda.amp.autocast():
            optimizer.zero_grad()

            outputs = model(data)

            loss = loss_fn(outputs, target)
            loss.backward()

            optimizer.step()

            running_loss = running_loss + loss.item()
        loop.set_postfix(loss = loss)

    print(running_loss/len(loader))
        # loop.set_postfix(loss=loss)


def main():
    if "dress.pth.tar" in os.listdir():
        model = torch.load("dress.pth.tar").to(DEVICE)
    else:
        model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    train_loader = get_loader(TRAIN_IMG_DIR, TRAIN_MASK_DIR, BATCH_SIZE)  #removed transformations

    for epoch in range(NUM_EPOCHS):
        print("Epochs: ", epoch + 1)
        train(train_loader, model, optimizer, loss_fn)
        if (epoch + 1)%5 == 0:
            save_checkpoint(model)
    save_checkpoint(model)

if __name__ == "__main__":
    main()