import torch
from dataset import DressDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="dress.pth.tar"):
    print("=> Saving Checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loader(
    train_dir,
    train_maskdir,
    batch_size,
    train_transform=None,
    num_workers=4,
    pin_memory=True,):

        train_ds = DressDataset(image_directory=train_dir, mask_directory=train_maskdir)

        train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)

        return train_loader