#modification : remove z_avg from feature vectors...

import torch
import torch.nn as nn
import torch.optim as optim
#from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import torchvision.transforms as transforms
from PIL import Image
from kmeans_pytorch import kmeans
import numpy as np
import argparse
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.stats import mode
from data import get_breast_data
from torch.utils.tensorboard import SummaryWriter


torch.backends.cudnn.enabled = True

device = 'cuda'

def parse_args():
    parser = argparse.ArgumentParser(description= 'breast ultrasound - CoCo')
    parser.add_argument('--dataset'     , default='BU',        help='BU/BT')
    parser.add_argument('--model'       , default='ResNet18',      help='ResNet{10|18|34|50|101}') # 50 and 101 are not used in the paper#relationnet_softmax replace L2 norm with softmax to expedite training, maml_approx use first-order approximation in the gradient for efficiency
    
    parser.add_argument('--num_classes' , default=3, type=int, help='total number of classes in softmax, only used in baseline') #make it larger than the maximum label value in base class
    parser.add_argument('--num_concepts' , default=64, type=int, help='total number of classes in softmax, only used in baseline') #make it larger than the maximum label value in base class
    parser.add_argument('--im_size' , default=256, type=int, help='image resolution') #make it larger than the maximum label value in base class
    parser.add_argument('--batch_size' , default=32, type=int, help='batch size') #make it larger than the maximum label value in base class
    # parser.add_argument('--batch_size' , default=32, type=int, help='batch size') #make it larger than the maximum label value in base class
    
    parser.add_argument('--save_freq'   , default=5, type=int, help='Save frequency')
    parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')        
    parser.add_argument('--stop_epoch'  , default=-1, type=int, help ='Stopping epoch')
    parser.add_argument('--resume'      , action='store_true', help='continue from previous trained model with largest epoch')
    parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want 
    parser.add_argument('--save_iter', default=-1, type=int,help ='save feature from the model trained in x epoch, use the best model if x is -1')

    return parser.parse_args()

def k_means_compute(embeddings, flag = None):

    X = embeddings
    # Initialize k-means with k=2
    cluster_ids_x, cluster_centers = kmeans(
    X=X, num_clusters=2, distance='euclidean', device=device)
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


def proto(datas, loader, model):

    # globalpool = nn.AdaptiveAvgPool2d((1,1))

    embeddings_b = []
    embeddings_m = []
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
                for j in range(args.num_concepts):
                    x,y = int(j%8), int(j/8)
                    feat.append(feature_maps[i, :, x, y])
                # feat.append(feature_avg[i, :])
                feat = torch.cat(feat, dim=0)
                feat_list.append(feat.view(1, -1))
            z_all = torch.cat(feat_list, dim=0)
            #---------------------
            embeddings_b.append(z_all)
            embeddings_m.append(z_all)
            embeddings_n.append(z_all)
        
        embeddings_b, embeddings_m, embeddings_n= torch.cat(embeddings_b,dim=0).to(device), torch.cat(embeddings_m,dim=0).to(device), torch.cat(embeddings_n,dim=0).to(device)
        labels_tensor = torch.cat(total_labels, dim=0).flatten()
        #------------------------
        mask_b = torch.eq(torch.zeros_like(labels_tensor), labels_tensor)
        mask_m = torch.eq(torch.zeros_like(labels_tensor)+1, labels_tensor)
        mask_n = torch.eq(torch.zeros_like(labels_tensor)+2, labels_tensor)
        #-----------------------
        embeddings_b = embeddings_b[mask_b].reshape(-1,512)
        embeddings_m = embeddings_m[mask_m].reshape(-1,512)
        embeddings_n = embeddings_n[mask_n].reshape(-1,512)
        #----------------------
        # print("k-means b computing...")
        centroids_b = k_means_compute(embeddings_b)
        # print("k-means m computing...")
        centroids_m = k_means_compute(embeddings_m)
        # print("k-means n computing...")
        centroids_n = k_means_compute(embeddings_n)
        #-----------------------
        embeddings_b  = torch.cat((centroids_b[0],centroids_b[1]))
        embeddings_m = torch.cat((centroids_m[0],centroids_m[1]))
        embeddings_n  = torch.cat((centroids_n[0],centroids_n[1]))


        return embeddings_b, embeddings_m, embeddings_n, embeddings_b.reshape(2,-1), embeddings_m.reshape(2,-1), embeddings_n.reshape(2,-1)
    # if datas==11:
    #   with open('./train_path.txt', 'r') as f:
    #       data = f.readlines()
    # elif datas==22:
    #   with open('./vals_path.txt', 'r') as f:
    #       data = f.readlines()
            
    # image_paths = []
    # class_labels = []
    
    # for line in data:
    #     parts = line.strip().split(" ",1)
    #     image_paths.append(parts[1])
    #     if 'benign' in parts[1]:
    #         label = 0
    #         class_labels.append(label)
    #     elif 'malignant' in parts[1]:
    #         label = 1
    #         class_labels.append(label)
    #     elif 'normal' in parts[1]:
    #         label = 2
    #         class_labels.append(label)
            
    # embeddings_b = torch.zeros(512*65).to(device)
    # count_b = 0
    # embeddings_m = torch.zeros(512*65).to(device)
    # count_m = 0
    # embeddings_n = torch.zeros(512*65).to(device)
    # count_n = 0
    
    # for i, path in enumerate(image_paths):
    #     image = Image.open(path).convert('RGB')
    #     image_tensor = transform(image).to(device)
        
    #     embedding,feature_map = model1(image_tensor.unsqueeze(0), 'embedding').squeeze().detach()            
        
    #     if class_labels[i] == 0:
    #         for j in range(64):
    #             x,y = int((j)%8), int((j)//8)
    #             embeddings_b[i*512:(i+1)*512] += feature_map[i, : , x, y]
    #         embeddings_b[64*512:] = embedding    
    #         # embeddings_b += embedding
    #         count_b += 1
    #     if class_labels[i] == 1:
    #         for j in range(64):
    #             x,y = int((j)%8), int((j)//8)
    #             embeddings_m[i*512:(i+1)*512] += feature_map[i, : , x, y]
    #         embeddings_m[64*512:]+=embedding    
    #         # embeddings_m += embedding
    #         count_m += 1
    #     else:
    #         for j in range(64):
    #             x,y = int((j+1)%8), int((j+1)//8)
    #             embeddings_n[i*512:(i+1)*512] += feature_map[i, : , x, y]
    #         embeddings_n[64*512:] = embedding    
    #         # embeddings_n += embedding
    #         count_n += 1
            
    # mean_embedding_b = embeddings_b / count_b
    # mean_embedding_m = embeddings_m / count_m
    # mean_embedding_n = embeddings_n / count_n
    
    # return mean_embedding_b, mean_embedding_m, mean_embedding_n



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


class MeanDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,labels,embeddings, means_b, means_m,means_n,mean_embedding_b,mean_embedding_m,mean_embedding_n):
        batch_size = embeddings.shape[0]
        embedding_size = means_b.shape[1]
        num_parts = 2
        
        mean_embedding_b = mean_embedding_b.reshape(1, -1)
        mean_embedding_m = mean_embedding_m.reshape(1, -1)
        mean_embedding_n = mean_embedding_n.reshape(1, -1)
        
        distances_b = torch.cdist(embeddings, mean_embedding_b)
        distances_m = torch.cdist(embeddings, mean_embedding_m)
        distances_n = torch.cdist(embeddings, mean_embedding_n)
        
        distances_im = torch.stack([distances_b, distances_m, distances_n], dim=1)
        
        min_distances, val = torch.min(distances_im, dim=1)
        val=val.flatten() #.to(device)
        
        part_embeddings = embeddings.reshape(batch_size, -1, embedding_size)
        #---------------------------
        means_all = torch.stack([means_b, means_m, means_n], dim = 0)  
        dis = torch.norm(part_embeddings.unsqueeze(1) - means_all.unsqueeze(0), dim=-1)
        #---------------------------
        # d = torch.stack([distances_b, distances_m, distances_n], dim=1)
        same_class_mask = labels.view(-1, 1).to("cpu") == torch.arange(dis.shape[1])
        same_class_mask = same_class_mask.unsqueeze(2).expand(-1, -1, num_parts)
        #------------------
        different_class_mask = ~ same_class_mask
        #---------------------------
        margin = 1
        same_concept_dists = dis[same_class_mask].reshape(-1) #same class
        diff_concept_dists = dis[different_class_mask].reshape(-1) #diff class
        zero = torch.zeros_like(same_concept_dists)
        loss= torch.max(zero, margin + same_concept_dists - diff_concept_dists.mean()).mean()
        
        return distances_im, val,loss


def compute_means(train_compute, model, num_concepts):

    #model=ResNet18Embedding(64)
    #num_ftrs = model.fc.in_features
    #model.fc = nn.Linear(num_ftrs, 2)
    #model.to(device)
    means_b = [None] * (num_concepts+1)
    means_m = [None] * (num_concepts+1)
    means_n = [None] * (num_concepts+1)
    
    count_b = 0
    count_m = 0
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
                        if means_b[part_id-1] is None:
                            means_b[part_id-1] = embedding.new_zeros(embedding.shape[0])
                        means_b[part_id-1] += embedding
                        count_b += 1
                        
                elif labels==1:
                    if part_id < num_concepts + 1:
                        if means_m[part_id-1] is None:
                            means_m[part_id-1] = embedding.new_zeros(embedding.shape[0])
                        means_m[part_id-1] += embedding
                        count_m += 1
                        
                elif labels==2:
                    if part_id < num_concepts + 1:
                        if means_n[part_id-1] is None:
                            means_n[part_id-1] = embedding.new_zeros(embedding.shape[0])
                        means_n[part_id-1] += embedding
                        count_n += 1
            #--------------------------------------------
            means_b[64]=embedding.new_zeros(embedding.shape[0])  
            means_m[64]=embedding.new_zeros(embedding.shape[0])  
            means_n[64]=embedding.new_zeros(embedding.shape[0])  
            #--------------------------------------------            
            means_b[64]+=embeddings[i] if labels == 0 else 0
            means_m[64]+=embeddings[i] if labels == 1 else 0
            means_n[64]+=embeddings[i] if labels == 2 else 0 
            #--------------------------------------------
            #   
    for i in range(num_concepts+1):
        if means_b[i] is not None:
            means_b[i] /= count_b
        if means_m[i] is not None:
            means_m[i] /= count_m
        if means_n[i] is not None:
            means_n[i] /= count_n
            
    means_b = torch.stack([mean.clone().detach() for mean in means_b])
    means_m = torch.stack([mean.clone().detach() for mean in means_m])
    means_n = torch.stack([mean.clone().detach() for mean in means_n])
    
    return means_b, means_m, means_n

# class ResNet18Embedding(nn.Module):
#     def __init__(self, embedding_size):
#         super(ResNet18Embedding, self).__init__()
#         resnet = resnet18(pretrained=False)
#         modules = list(resnet.children())[:-1]      # Remove last layer
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

def train(args, tf_writer):

    train_dataloader, val_dataloader = get_breast_data(args.im_size, args.batch_size)
    
    mod=ResNet18Embedding(args.num_concepts)
    # num_ftrs = mod.fc.in_features
    # mod.fc = nn.Linear(num_ftrs, args.num_concepts) #args.num_classes
    mod.to(device)
    #--------------------
    criterion = MeanDistanceLoss()
    #--------------------
    optimizer = optim.Adam(mod.parameters(), lr=0.001)
    max_acc=0
    max_acc_mod=0
    
    for epoch in range(20):
        print("Training Epoch", epoch)
        running_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        correct=0
        
        for batch in train_dataloader:
            # centroids = batch['centroids']
            labels = batch['labels'].to(device)
            
            embeddings = parse_feature(batch, mod, args.num_concepts)
    
            #print('emb size', embeddings.shape, labels.shape)
            # loss_g = F.cross_entropy(embeddings, labels) # saeed : why????!
            print("Computing Prototypes(Training)...")
            mean_embedding_b, mean_embedding_m, mean_embedding_n, means_b, means_m, means_n = proto(11, train_dataloader, mod)
    
            print("Computing Concept Prototypes(Training)...")
            # means_b, means_m, means_n = compute_means(train_dataloader, mod, args.num_concepts)
            #mean_embedding_b, mean_embedding_m, mean_embedding_n=Variable(mean_embedding_b), Variable(mean_embedding_m), Variable(mean_embedding_n)
            
            distances, val, loss = criterion(labels,embeddings, means_b,means_m, means_n, mean_embedding_b, mean_embedding_m,mean_embedding_n)            
            
            #print(distances.min(), distances.max(), embeddings.min(), embeddings.max(), labels.min(), labels.max())
            #print(distances.shape)
            loss_p = F.cross_entropy(-distances.to(device), labels.unsqueeze(-1))
            
            losstt = loss +  loss_p 
            
            print("Loss: ", losstt.item())
            
            # losstt.requires_grad = True
            optimizer.zero_grad()
            losstt.backward()
            optimizer.step()
            
            running_loss += losstt.item()
            
            #labels=labels.to('cpu')
            batch_correct = torch.eq(val.to(device), labels).sum().item() #.to('cpu')
            epoch_correct += batch_correct
            epoch_total += len(labels)
            # _, predicted = torch.max(embeddings.data, 1)
            #predicted=predicted.to('cpu')
            # correct += (predicted == labels).sum().item()

        accuracy_model = 100 * epoch_correct / epoch_total
        tf_writer.add_scalar('loss/train', running_loss, epoch)
        tf_writer.add_scalar('acc/train', accuracy_model, epoch)
        
        print('Epoch {}: Accuracy(train set) = {:.2f}%'.format(epoch, accuracy_model))

        #criterion = MeanDistanceLoss()
        epoch_total_val = 0
        epoch_correct_val=0
        for batch in val_dataloader:
            images = batch['image'].to(device)
            # centroids = batch['centroids']
            labels = batch['labels'].to(device)
            
            with torch.no_grad():
                embeddings = parse_feature(batch, mod, args.num_concepts)

            
            
            print("Computing Prototypes(val)...")
            mean_embedding_b, mean_embedding_m, mean_embedding_n,_,_,_ = proto(11, val_dataloader, mod)
            #-----------------------------------
            distances_b = torch.cdist(embeddings, mean_embedding_b.reshape(1, -1))
            distances_m = torch.cdist(embeddings, mean_embedding_m.reshape(1, -1))
            distances_n = torch.cdist(embeddings, mean_embedding_n.reshape(1, -1))
            #-----------------------------------
            distances_im = torch.stack([distances_b, distances_m, distances_n], dim=1)
            #-----------------------------------
            min_distances, val = torch.min(distances_im, dim=1)
            val = val.view(-1)
            batch_correct_val= torch.eq(val.to(device), labels).sum().item() #.to('cpu')
            epoch_correct_val += batch_correct_val
            epoch_total_val += len(labels)
            
        #---------------------------
        accuracy_model_val = 100 * epoch_correct_val / epoch_total_val   
        tf_writer.add_scalar('acc/test', accuracy_model_val, epoch)


        if accuracy_model_val > max_acc : 
            max_acc = accuracy_model_val
            outfile = 'best_model_model_v4_val.tar'
            torch.save({'epoch':epoch, 'state':mod.state_dict()}, outfile)
        if accuracy_model > max_acc_mod : 
            print("best model for model! save...")
            max_acc_mod = accuracy_model
            outfile = 'best_model_model_v4_train.tar'
            torch.save({'epoch':epoch, 'state':mod.state_dict()}, outfile)

        if (epoch % 5==0) or (epoch==10-1):
            # outfile = ('{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':mod.state_dict()}, 'resnet18_model_v4.pth')
            
        print('Epoch {}: Accuracy(test set) = {:.2f}%'.format(epoch, accuracy_model_val))


if __name__ == '__main__':
    args = parse_args()
     
    print("Training Started...")
    tf_writer = SummaryWriter(log_dir="./runs/run1_debug_28aug_v4")
    train(args, tf_writer)