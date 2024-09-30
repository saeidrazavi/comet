from kmeans_pytorch import kmeans
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from PIL import Image
import yaml

device = 'cuda'

def k_means_compute(embeddings, flag = None):

    X = embeddings
    # Initialize k-means with k=2
    cluster_ids_x, cluster_centers = kmeans(
    X=X, num_clusters=2, distance='euclidean', tol=1e-7, device=device)
    # kmeans = KMeans(n_clusters=2, init='random', n_init=10, max_iter=20, random_state=42)

    # Count the number of points in each cluster
    unique, counts = torch.unique(cluster_ids_x, return_counts=True)
    cluster_counts = dict(zip(unique.tolist(), counts.tolist()))
    # Identify the cluster with the maximum data points
    max_cluster_id = max(cluster_counts, key=cluster_counts.get)
    # Check if max_cluster_id is 1 and swap centroids if necessary
    if max_cluster_id == 1:
        # Swap the centroids at index 0 and 1
        cluster_centers[0], cluster_centers[1] = cluster_centers[1].clone(), cluster_centers[0].clone()

    if flag == "parse_feature" :     
       
        embeddings  = torch.cat((cluster_centers[0],cluster_centers[1]))
        return embeddings 
    
    else : 
        return cluster_centers



def parse_feature(batch, model, num_concepts):
        # globalpool = nn.AdaptiveAvgPool2d((1,1))
        feature_maps = model(batch['image'].to(device), 'feature_map')
        # feature_avg = globalpool(feature_maps).view(feature_maps.size(0), feature_maps.size(1))
        batch_num = feature_maps.size(0)

        feat_list = []
        for i in range(batch_num):
            feat = []
            for j in range(num_concepts):
                x,y = int(j%8), int(j/8)
                feat.append(feature_maps[i, :, x, y])
            # feat.append(feature_avg[i, :])
            feat = torch.cat(feat, dim=0).reshape(64,-1)
            feat = k_means_compute(feat, 'parse_feature')
            feat_list.append(feat.view(1, -1))

        z_all = torch.cat(feat_list, dim=0)
        #-----------
        return z_all


def proto(datas, loader, model, num_concepts=64):

    # globalpool = nn.AdaptiveAvgPool2d((1,1))

    embeddings_y = []
    embeddings_n = []
    total_labels = []
    
    if datas == 11:  
        
        for batch in loader: 
                
            feature_maps = model(batch['image'].to(device), 'feature_map')
            #----------------
            labels = batch['labels']
            total_labels.append(labels)
            #----------------
            # feature_avg = globalpool(feature_maps).view(feature_maps.size(0), feature_maps.size(1))
            batch_num = feature_maps.size(0)

            feat_list = []
            for i in range(batch_num):
                feat = []
                for j in range(num_concepts):
                    x,y = int(j%8), int(j/8)
                    feat.append(feature_maps[i, :, x, y])
                # feat.append(feature_avg[i, :])
                feat = torch.cat(feat, dim=0)
                feat_list.append(feat.view(1, -1))
            z_all = torch.cat(feat_list, dim=0)
            #---------------------
            embeddings_y.append(z_all)
            embeddings_n.append(z_all)
        
        embeddings_y, embeddings_n= torch.cat(embeddings_y,dim=0).to(device), torch.cat(embeddings_n,dim=0).to(device)
        labels_tensor = torch.cat(total_labels, dim=0).flatten()
        #------------------------
        mask_y = torch.eq(torch.zeros_like(labels_tensor), labels_tensor)
        mask_n = torch.eq(torch.zeros_like(labels_tensor)+1, labels_tensor)
        #-----------------------
        embeddings_y = embeddings_y[mask_y].reshape(-1,512)
        embeddings_n = embeddings_n[mask_n].reshape(-1,512)
        #----------------------
        # print("k-means b computing...")
        centroids_y = k_means_compute(embeddings_y)
        # print("k-means n computing...")
        centroids_n = k_means_compute(embeddings_n)
        #-----------------------
        embeddings_y  = torch.cat((centroids_y[0],centroids_y[1]))
        embeddings_n  = torch.cat((centroids_n[0],centroids_n[1]))


        return embeddings_y, embeddings_n, embeddings_y.reshape(2,-1), embeddings_n.reshape(2,-1)


def compute_means(train_compute, model, num_concepts):
       
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
                    # if x<20 or y<20:
                    #          crop =  image_i[:, 0:(x+40), 0:(y+40)]
                    # else:
                    #          crop =  image_i[:, (x-20):(x+20), (y-20):(y+20)]
                            
                    # crop = crop.unsqueeze(0)   
                    
                    # with torch.no_grad():
                    #     embedding=model(crop).squeeze(0)
                        
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
         
        means_y = torch.stack([mean.clone().detach() for mean in means_y])
        means_n = torch.stack([mean.clone().detach() for mean in means_n])
      
        return means_y, means_n        


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


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)
