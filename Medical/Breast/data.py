from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

class BreastDataset(Dataset):
    def __init__(self, train_ids_file, centroids_file, transform=None):
        self.image_paths = {}
        self.train_ids = []
        with open(train_ids_file) as f:
            for line in f:
                image_id, image_path = line.strip().split(" ",1)
                self.image_paths[image_id] = image_path
                self.train_ids.append(image_id)

        self.centroids = {}
        with open(centroids_file) as f:
            part_ids = [str(i+1) for i in range(64)]
            for line in f:
                img_id, *parts = map(int, line.strip().split())
                if img_id not in self.centroids:
                    self.centroids[img_id] = {'part_ids': part_ids, 'x': [], 'y': []}
                self.centroids[img_id]['x'].extend(parts[1::2])
                self.centroids[img_id]['y'].extend(parts[2::2])
        self.transform = transform

    def __len__(self):
        return len(self.train_ids)

    def __getitem__(self, index):
        image_id = self.train_ids[index]
        if image_id not in self.image_paths:
            raise ValueError(f"Image id {image_id} not found in image paths")
        image_path = self.image_paths[image_id]
        image = Image.open(image_path).convert('RGB')
        centroids = self.centroids[int(image_id)]
        if 'benign' in image_path:
            label = 0
        elif 'malignant' in image_path:
            label = 1
        elif 'normal' in image_path:
            label = 2
        else:
            raise ValueError(f"Unknown label for image at {image_path}")    
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'image_id': image_id, 'centroids': centroids, 'labels': label}
    
    
def get_breast_data(im_size, batch_size, flag = None):
    train_transform = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.RandomHorizontalFlip(0.2),
        transforms.RandomVerticalFlip(0.1),
        transforms.RandomAutocontrast(0.2),
        transforms.ToTensor()
    ])

    train_transform_v2 = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor()
    ]) 

    if flag is None :  
        train_dataset = BreastDataset('./train_path.txt', './centroid_train.txt', transform=train_transform_v2)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        val_dataset = BreastDataset('./vals_path.txt', './centroid_val.txt', transform=train_transform_v2)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        return train_dataloader, val_dataloader

    else : 

        test_dataset = BreastDataset('./test_path.txt', './centroid_tester.txt', transform=train_transform)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        return test_dataloader

