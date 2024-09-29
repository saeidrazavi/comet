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


def proto(datas):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.2),
        transforms.ToTensor()
    ])
    model1=ResNet18Embedding(64)
    num_ftrs = model1.fc.in_features
    model1.fc = nn.Linear(num_ftrs, 2)
    modi= torch.load('./best_model_model.tar')
    model1.load_state_dict(modi['state'])
    model1.cuda()
    if datas==11:
      with open('./train_path.txt', 'r') as f:
          data = f.readlines()
    elif datas==33:
      with open('./val_path.txt', 'r') as f:
          data = f.readlines()
    image_paths = []
    class_labels = []
    for line in data:
        parts = line.strip().split(" ",1)
        image_paths.append(parts[1])
        if 'yes' in parts[1]:
            label = 0
            class_labels.append(label)
        elif 'no' in parts[1]:
            label = 1
            class_labels.append(label)
    embeddings_y = torch.zeros(2).cuda()
    count_y = 0
    embeddings_n = torch.zeros(2).cuda()
    count_n = 0
    for i, path in enumerate(image_paths):
        image = Image.open(path).convert('RGB')
        image_tensor = transform(image).cuda()
        embedding = model1(image_tensor.unsqueeze(0)).squeeze()
        if class_labels[i] == 0:
            embeddings_y += embedding
            count_y += 1
        elif class_labels[i] == 1:
            embeddings_n += embedding
            count_n += 1
    mean_embedding_y = embeddings_y / count_y
    mean_embedding_n = embeddings_n / count_n
    return mean_embedding_y, mean_embedding_n

class BrainDataset(Dataset):
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
        if 'yes' in image_path:
            label = 0
        elif 'no' in image_path:
            label = 1
        else:
            raise ValueError(f"Unknown label for image at {image_path}")    
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'image_id': image_id, 'centroids': centroids, 'labels': label}

class MeanDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,labels, model,images,embeddings, means_y,means_n, centroids,mean_embedding_y,mean_embedding_n):
        mean_embedding_y = mean_embedding_y.reshape(1, -1).to('cuda')
        mean_embedding_n = mean_embedding_n.reshape(1, -1).to('cuda')
        distances_y = torch.cdist(embeddings, mean_embedding_y).to('cuda')
        distances_n = torch.cdist(embeddings, mean_embedding_n).to('cuda')
        distances1 = torch.stack([distances_y, distances_n], dim=1).to('cuda')
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
        distances_y = torch.cdist(part_embeddings.to('cuda'), means_y.unsqueeze(0).to('cuda'))
        distances_n = torch.cdist(part_embeddings.to('cuda'), means_n.unsqueeze(0).to('cuda'))
        t,n = [], []
        t_distances, n_distances  = [], []
        for i in range(distances_y.size(0)):
            t_i = []
            n_i = []
            t_distances_i = []
            n_distances_i = []
            for j in range(distances_y.size(1)):
                if distances_y[i,j,j] < distances_n[i,j,j]:
                    t_i.append(j)
                    t_distances_i.append(distances_y[i,j,j].tolist())
                else:
                    n_i.append(j)
                    n_distances_i.append(distances_n[i,j,j].tolist())
            t.append(t_i)
            n.append(n_i)
            t_distances.extend(t_distances_i)
            n_distances.extend(n_distances_i)
        d=torch.stack([distances_y, distances_n], dim=1)
        closest_class = torch.argmin(torch.stack([distances_y, distances_n], dim=1), dim=1)
        batch_size = part_embeddings.size(0)
        num_parts = part_embeddings.size(1)
        mask = torch.eye(num_parts).repeat(batch_size, 1, 1)
        distances = torch.cdist(part_embeddings, part_embeddings)
        same_class_distances = distances * mask
        different_class_distances = distances * (1 - mask)
        same_class_distances=same_class_distances.to('cuda')
        different_class_distances=different_class_distances.to('cuda')
        mean_y_same = torch.sum(same_class_distances[closest_class == 0]) / torch.sum(closest_class == 0)
        mean_y_different = torch.sum(different_class_distances[closest_class == 0]) / torch.sum(closest_class == 0)
        mean_n_same = torch.sum(same_class_distances[closest_class == 1]) / torch.sum(closest_class == 1)
        mean_n_different = torch.sum(different_class_distances[closest_class == 1]) / torch.sum(closest_class == 1)
        mean_y_same=mean_y_same.to('cuda')
        mean_n_same=mean_n_same.to('cuda')
        mean_y_different=mean_y_different.to('cuda')
        mean_n_different=mean_n_different.to('cuda')
        margin = 0.5
        loss_same = torch.max(torch.zeros(1).to('cuda'), mean_y_same +mean_n_same - 2 * margin)
        loss_different = torch.max(torch.zeros(1).to('cuda'), 2 * margin - mean_y_different - mean_n_different)
        loss = loss_same + loss_different
        mode_class, _ = torch.mode(closest_class, dim=1)
        mode_class_list = mode_class.tolist()
        mode_class_per_image = [max(set(lst), key=lst.count) for lst in mode_class_list]
        return distances1, val,loss, mode_class_per_image


def compute_means(data, num_concepts):
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(0.2),
        transforms.ToTensor()
    ])

    if data==33:
      test_dataset = BrainDataset('/content/test_image_paths.txt', '/content/centroids_test.txt', transform=test_transform)
      test_compute = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    model=ResNet18Embedding(64)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 64)
    modi= torch.load('./best_model_model.tar')
    model.load_state_dict(modi['state'])
    model.cuda()
    #-----------------------
    
    means_y = [None] * (num_concepts+1)
    means_n = [None] * (num_concepts+1)
    
    count_y = 0
    count_n = 0
    
    for batch in train_compute:

        embeddings, feature_maps = model(batch['image'].to(device))
        # images= batch['image'].to(device)
        
        for i in range(feature_maps.shape[0]):
            # image_i = images[i]
            # image_id = int(batch['image_id'][i])
            centroids = batch['centroids']
            labels=batch['labels'][i]
            
            part_ids = [int(t[i]) for t in centroids['part_ids']]
        
            for part_id in part_ids:
                x, y = int((part_id-1)%8), int((part_id-1)/8)
                embedding = feature_maps [i, :, x, y]
                    
                if labels==0:
                    if part_id < num_concepts + 1:
                        if means_y[part_id-1] is None:
                            means_y[part_id-1] = embedding.new_zeros(embedding.shape[0])
                        means_y[part_id-1] += embedding
                        count_y += 1
                        
                elif labels==1:
                    if part_id < num_concepts + 1:
                        if means_n[part_id-1] is None:
                            means_n[part_id-1] = embedding.new_zeros(embedding.shape[0])
                        means_n[part_id-1] += embedding
                        count_n += 1
                        
            #--------------------------------------------
            means_y[64]=embedding.new_zeros(embedding.shape[0])  
            means_n[64]=embedding.new_zeros(embedding.shape[0])   
            #--------------------------------------------            
            means_y[64]+=embeddings[i] if labels == 0 else 0
            means_n[64]+=embeddings[i] if labels == 1 else 0
            #--------------------------------------------
            #   
    for i in range(num_concepts+1):
        if means_y[i] is not None:
            means_y[i] /= count_y
        if means_n[i] is not None:
            means_n[i] /= count_n
        
    means_y = torch.stack([mean.clone().detach() for mean in means_y]).cuda()
    means_n = torch.stack([mean.clone().detach() for mean in means_n]).cuda()
    
    return means_y, means_n


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
class ResNet18Embedding(nn.Module):
    def __init__(self, embedding_size):
        super(ResNet18Embedding, self).__init__()
        resnet = resnet18(pretrained=False)
        # Layer up to Layer 4
        self.layer4 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        # Average pooling and fully connected layer
        self.avgpool = resnet.avgpool
        
    def forward(self, x, type=None):
        # Pass input through layers up to Layer 4
        x_prime = self.layer4(x)
        # Pass through average pooling
        if type == 'feature_map':
           
           return x_prime
        
        if type == 'embedding':
           
           x = self.avgpool(x_prime)
           # Flatten and pass through the fully connected layer
           x = x.view(x.size(0), -1)
        #    x = self.fc(x)
           return x 
        
        else : 
            x = self.avgpool(x_prime)
            # Flatten and pass through the fully connected layer
            x = x.view(x.size(0), -1)
            # x = self.fc(x)
            return x, x_prime
        # Return both the embedding and the feature maps at Layer 4


def test(mean_embedding_y, mean_embedding_n):
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.2),
        transforms.ToTensor()
    ])
    test_dataset = BrainDataset('/content/test_image_paths.txt', '/content/centroids_test.txt', transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    criterion = MeanDistanceLoss()
    mod=ResNet18Embedding(64)
    num_ftrs = mod.fc.in_features
    mod.fc = nn.Linear(num_ftrs, 2)
    mod.cuda()
    modi= torch.load('/content/best_model_model.tar')
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
            means_y, means_n = compute_means(33)
            mean_embedding_y, mean_embedding_n=Variable(mean_embedding_y), Variable(mean_embedding_n)
            distances, val,loss,im = criterion(labels,mod,images,embeddings, means_y, means_n, centroids, mean_embedding_y,mean_embedding_n)            
            labels=torch.tensor(labels)
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
        epoch_accuracy = 100 * epoch_correct / epoch_total
        print('Accuracy (Model)= {:.2f}%'.format(accuracy_model))


if __name__ == '__main__':
    mean_embedding_y, mean_embedding_n=proto(33)
    test(mean_embedding_y, mean_embedding_n)