from torchvision import datasets, transforms
from torch.utils.data import Dataset
import random
from collections import defaultdict

class LimitedPerClassDataset(Dataset):
    def __init__(self, image_folder, limit=30):
        targets = image_folder.targets
        samples = image_folder.samples
        class_indices = defaultdict(list)
        for idx, label in enumerate(targets):
            class_indices[label].append(idx)
        selected_indices = []
        for label, idxs in class_indices.items():
            if len(idxs) > limit:
                idxs = random.sample(idxs, limit)
            selected_indices.extend(idxs)
        self.samples = [samples[i] for i in selected_indices]
        self.targets = [targets[i] for i in selected_indices]
        self.classes = image_folder.classes
        self.class_to_idx = image_folder.class_to_idx
        self.transform = image_folder.transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        from PIL import Image
        sample = Image.open(path).convert('RGB')
        if self.transform:
            sample = self.transform(sample)
        return sample, target

def get_data_loaders(data_dir, img_size, batch_size, limit_per_class=30):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    base_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    full_dataset = LimitedPerClassDataset(base_dataset, limit=limit_per_class)
    # train/val/test split
    from torch.utils.data import random_split, DataLoader
    train_len = int(len(full_dataset) * 0.9)
    val_len = int(len(full_dataset) * 0.05)
    test_len = len(full_dataset) - train_len - val_len
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_len, val_len, test_len])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader, full_dataset.classes
