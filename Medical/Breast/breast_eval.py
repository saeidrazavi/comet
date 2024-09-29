import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.stats import mode
from breast_train import ResNet18Embedding


def proto(datas):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.2),
        transforms.ToTensor()
    ])
    model1=ResNet18Embedding(64)
    num_ftrs = model1.fc.in_features
    model1.fc = nn.Linear(num_ftrs, 64)
    modi= torch.load('./best_model_model.tar')
    model1.load_state_dict(modi['state'])
    model1.cuda()
    if datas==33:
      with open('./test_path.txt', 'r') as f:
          data = f.readlines()
    image_paths = []
    class_labels = []
    for line in data:
        parts = line.strip().split(" ",1)
        image_paths.append(parts[1])
        if 'benign' in parts[1]:
            label = 0
            class_labels.append(label)
        elif 'malignant' in parts[1]:
            label = 1
            class_labels.append(label)
        elif 'normal' in parts[1]:
            label = 2
            class_labels.append(label)
    embeddings_b = torch.zeros(64).cuda()
    count_b = 0
    embeddings_m = torch.zeros(64).cuda()
    count_m = 0
    embeddings_n = torch.zeros(64).cuda()
    count_n = 0
    for i, path in enumerate(image_paths):
        image = Image.open(path).convert('RGB')
        image_tensor = transform(image).cuda()
        embedding = model1(image_tensor.unsqueeze(0)).squeeze()
        if class_labels[i] == 0:
            embeddings_b += embedding
            count_b += 1
        if class_labels[i] == 1:
            embeddings_m += embedding
            count_m += 1
        else:
            embeddings_n += embedding
            count_n += 1
    mean_embedding_b = embeddings_b / count_b
    mean_embedding_m = embeddings_m / count_m
    mean_embedding_n = embeddings_n / count_n
    return mean_embedding_b, mean_embedding_m, mean_embedding_n

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

class MeanDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,labels, model,images,embeddings, means_b, means_m,means_n, centroids,mean_embedding_b,mean_embedding_m,mean_embedding_n):
        mean_embedding_b = mean_embedding_b.reshape(1, -1).to('cuda') 
        mean_embedding_m = mean_embedding_m.reshape(1, -1).to('cuda')
        mean_embedding_n = mean_embedding_n.reshape(1, -1).to('cuda')
        distances_b = torch.cdist(embeddings, mean_embedding_b).to('cuda')
        distances_m = torch.cdist(embeddings, mean_embedding_m).to('cuda')
        distances_n = torch.cdist(embeddings, mean_embedding_n).to('cuda')
        distances1 = torch.stack([distances_b, distances_m, distances_n], dim=1).to('cuda')
        min_distances, val = torch.min (distances1, dim=1)
        val=val.flatten().to('cuda')
        batch_size = embeddings.shape[0]
        num_parts = 64
        embedding_size = embeddings.shape[1]
        part_embeddings = torch.zeros(batch_size, num_parts, embedding_size)
        for i in range(batch_size):
            xs = [int(t[i]) for t in centroids['x']]
            ys = [int(t[i]) for t in centroids['y']]
            for j in range(num_parts):
                x = xs[j]
                y = ys[j]
                if x < 20 or y < 20:
                    crop = images[i, :, 0:(x+40), 0:(y+40)]
                else:
                    crop = images[i, :, (x-20):(x+20), (y-20):(y+20)]
                with torch.no_grad():
                    part_embedding = model(crop.unsqueeze(0))               
                part_embeddings[i, j, :] = part_embedding.squeeze()
        distances_b = torch.cdist(part_embeddings.to('cuda'), means_b.unsqueeze(0).to('cuda'))
        distances_m = torch.cdist(part_embeddings.to('cuda'), means_m.unsqueeze(0).to('cuda'))
        distances_n = torch.cdist(part_embeddings.to('cuda'), means_n.unsqueeze(0).to('cuda'))
        d=torch.stack([distances_b, distances_m, distances_n], dim=1)
        closest_class = torch.argmin(torch.stack([distances_b, distances_m, distances_n], dim=1), dim=1)
        batch_size = part_embeddings.size(0)
        num_parts = part_embeddings.size(1)
        mask = torch.eye(num_parts).repeat(batch_size, 1, 1)
        distances = torch.cdist(part_embeddings, part_embeddings)
        same_class_distances = distances * mask
        different_class_distances = distances * (1 - mask)
        same_class_distances=same_class_distances.to('cuda')
        different_class_distances=different_class_distances.to('cuda')
        mean_b_same = torch.sum(same_class_distances[closest_class == 0]) / torch.sum(closest_class == 0)
        mean_b_different = torch.sum(different_class_distances[closest_class == 0]) / torch.sum(closest_class == 0)
        mean_m_same = torch.sum(same_class_distances[closest_class == 1]) / torch.sum(closest_class == 1)
        mean_m_different = torch.sum(different_class_distances[closest_class == 1]) / torch.sum(closest_class == 1)
        mean_n_same = torch.sum(same_class_distances[closest_class == 2]) / torch.sum(closest_class == 2)
        mean_n_different = torch.sum(different_class_distances[closest_class == 2]) / torch.sum(closest_class == 2)
        mean_b_same=mean_b_same.to('cuda')
        mean_n_same=mean_n_same.to('cuda')
        mean_m_same=mean_m_same.to('cuda')
        mean_b_different=mean_b_different.to('cuda')
        mean_m_different=mean_m_different.to('cuda')
        mean_n_different=mean_n_different.to('cuda')
        margin = 0.5
        loss_same = torch.max(torch.zeros(1).to('cuda'), mean_b_same + mean_m_same+mean_n_same - 2 * margin)
        loss_different = torch.max(torch.zeros(1).to('cuda'), 2 * margin - mean_b_different - mean_m_different - mean_n_different)
        loss = loss_same + loss_different
        mode_class, _ = torch.mode(closest_class, dim=1)
        mode_class_list = mode_class.tolist()
        mode_class_per_image = [max(set(lst), key=lst.count) for lst in mode_class_list]
        return distances1, val,loss, mode_class_per_image


def compute_means(data):
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(0.2),
        transforms.ToTensor()
    ])

    if data==33:
      test_dataset = BreastDataset('./test_path.txt', './centroid_tester.txt', transform=train_transform)
      train_compute=DataLoader(test_dataset, batch_size=156, shuffle=True, num_workers=4)

    model=ResNet18Embedding(64)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 64)
    modi= torch.load('./best_model_model.tar')
    model.load_state_dict(modi['state'])
    model.cuda()
    means_b = [None] * 64
    means_m = [None] * 64
    means_n = [None] * 64
    count_b = 0
    count_m = 0
    count_n = 0
    for batch in train_compute:
        embeddings,feature_maps = model(batch['image'].cuda())
        # images= batch['image'].cuda()
        for i in range(embeddings.shape[0]):
            # image_i = images[i]
            # image_id = int(batch['image_id'][i])
            centroids = batch['centroids']
            labels=batch['labels'][i]
            part_ids = [int(t[i]) for t in centroids['part_ids']]
            xs=[int(t[i]) for t in centroids['x']]
            ys=[int(t[i]) for t in centroids['y']]
            for part_id, x, y in zip(part_ids,xs,ys):
                x, y = int(x/28), int(y/28)
                # if x<20 or y<20:
                #          crop =  image_i[:, 0:(x+40), 0:(y+40)]
                # else:
                #          crop =  image_i[:, (x-20):(x+20), (y-20):(y+20)]
                # crop = crop.unsqueeze(0)   
                # with torch.no_grad():
                #   embedding=model(crop).squeeze(0)
                embedding = feature_maps[i, : , x, y]
                if labels==0:
                  if part_id < 65:
                      if means_b[part_id-1] is None:
                          means_b[part_id-1] = embedding.new_zeros(feature_maps.shape[1])
                      means_b[part_id-1] += embedding
                      count_b += 1
                elif labels==1:
                  if part_id < 65:
                      if means_m[part_id-1] is None:
                          means_m[part_id-1] = embedding.new_zeros(feature_maps.shape[1])
                      means_m[part_id-1] += embedding
                      count_m += 1
                elif labels==2:
                  if part_id < 65:
                      if means_n[part_id-1] is None:
                          means_n[part_id-1] = embedding.new_zeros(feature_maps.shape[1])
                      means_n[part_id-1] += embedding
                      count_n += 1
    for i in range(64):
        if means_b[i] is not None:
            means_b[i] /= count_b
        if means_m[i] is not None:
            means_m[i] /= count_m
        if means_n[i] is not None:
            means_n[i] /= count_n
    means_b = torch.stack([torch.tensor(mean) for mean in means_b]).to('cuda')
    means_m = torch.stack([torch.tensor(mean) for mean in means_m]).to('cuda')
    means_n = torch.stack([torch.tensor(mean) for mean in means_n]).to('cuda')
    return means_b, means_m, means_n

# class ResNet18Embedding(nn.Module):
#     def __init__(self, embedding_size):
#         super(ResNet18Embedding, self).__init__()
#         resnet = resnet18(pretrained=False)
#         modules = list(resnet.children())[:-1]      
#         self.resnet = nn.Sequential(*modules)
#         self.fc = nn.Linear(resnet.fc.in_features, embedding_size)
    
#     def forward(self, x):
#         x = self.resnet(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x


def test(mean_embedding_b, mean_embedding_m, mean_embedding_n):
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.2),
        transforms.RandomVerticalFlip(0.1),
        transforms.RandomAutocontrast(0.2),
        transforms.ToTensor()
    ])
    test_dataset = BreastDataset('./test_path.txt', './centroid_tester.txt', transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)
    criterion = MeanDistanceLoss()
    mod=ResNet18Embedding(64)
    num_ftrs = mod.fc.in_features
    mod.fc = nn.Linear(num_ftrs, 64)
    mod.cuda()
    modi= torch.load('./best_model_model.tar')
    mod.load_state_dict(modi['state'])
    for epoch in range(1):
        running_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        correct=0
        for batch in test_dataloader:
            images = batch['image'].cuda()
            centroids = batch['centroids']
            labels = batch['labels']
            labels=labels.to('cuda')
            with torch.no_grad():
              embeddings = mod(images)
            means_b,means_m, means_n = compute_means(33)
            mean_embedding_b, mean_embedding_m, mean_embedding_n=Variable(mean_embedding_b), Variable(mean_embedding_m), Variable(mean_embedding_n)
            distances, val,loss,im= criterion(labels,mod,images,embeddings, means_b,means_m, means_n, centroids, mean_embedding_b, mean_embedding_m,mean_embedding_n)
            labels=torch.tensor(labels).to('cuda')
            labels=labels.type(torch.LongTensor).to('cpu')
            im=torch.tensor(im).to('cpu')
            batch_correct = torch.eq(val.to('cpu'), labels).sum().item()
            epoch_correct += batch_correct
            epoch_total += len(labels)
            embeddings=embeddings.to('cuda')
            _, predicted = torch.max(embeddings.data, 1)
            predicted=predicted.to('cpu')
            correct += (predicted == labels).sum().item()
        accuracy_model = 100 * correct / epoch_total
        print('Accuracy (Model)= {:.2f}%'.format(accuracy_model))

if __name__ == '__main__':
     mean_embedding_b, mean_embedding_m, mean_embedding_n=proto(33)
     test(mean_embedding_b, mean_embedding_m, mean_embedding_n)