import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import torchvision.transforms as transforms
from PIL import Image
import argparse
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from kmeans_pytorch import kmeans
from scipy.stats import mode
from torch.utils.tensorboard import SummaryWriter
from kmeans import lloyd

device = 'cuda'


def parse_args():
    parser = argparse.ArgumentParser(description= 'breast ultrasound - CoCo')
    parser.add_argument('--model'       , default='ResNet18',      help='ResNet{10|18|34|50|101}') # 50 and 101 are not used in the paper#relationnet_softmax replace L2 norm with softmax to expedite training, maml_approx use first-order approximation in the gradient for efficiency
    parser.add_argument('--num_classes' , default=3, type=int, help='total number of classes in softmax, only used in baseline') #make it larger than the maximum label value in base class
    parser.add_argument('--num_concepts' , default=64, type=int, help='total number of classes in softmax, only used in baseline') #make it larger than the maximum label value in base class
    parser.add_argument('--im_size' , default=256, type=int, help='image resolution') #make it larger than the maximum label value in base class
    parser.add_argument('--batch_size' , default=4, type=int, help='batch size') #make it larger than the maximum label value in base class
    # parser.add_argument('--batch_size' , default=32, type=int, help='batch size') #make it larger than the maximum label value in base class
    
    parser.add_argument('--save_freq'   , default=5, type=int, help='Save frequency')
    parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')        
    parser.add_argument('--stop_epoch'  , default=-1, type=int, help ='Stopping epoch')
    parser.add_argument('--save_iter', default=-1, type=int,help ='save feature from the model trained in x epoch, use the best model if x is -1')

    return parser.parse_args()


# Define function to get last 4 ReLU activations for the pretrained model
def get_last_4_relu_pretrained(model, x):
     
    activations = []
    hooks = []

    def getActivation():
    # the hook signature
        def hook(model, input, output):
            activations.append(output.detach())
        return hook
    
    layers_to_hook = [
        model.layer4[1].relu,  # ReLU after the second block in layer 4
        model.layer4[0].relu,  # ReLU after the first block in layer 4
        model.layer3[1].relu,  # ReLU after the second block in layer 3
        model.layer3[0].relu   # ReLU after the first block in layer 3
    ]


    for layer in layers_to_hook:
        hook = layer.register_forward_hook(getActivation())
        hooks.append(hook)

    # Forward pass to capture activations
    out = model(x)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return activations
    
def get_last_4_relu_scratch(model, x):

    activations = []
    hooks = []
    def getActivation():
    # the hook signature
        def hook(model, input, output):
            activations.append(output.detach())
        return hook
    
    layers_to_hook = [
       # ReLU after first block in layer 4
        model.layer4[6][0].relu,  # ReLU after second block in layer 3
        model.layer4[6][1].relu,  # ReLU after first block in layer 3
        model.layer4[7][0].relu,  # ReLU after second block in layer 4
        model.layer4[7][1].relu, 
    ]


    for layer in layers_to_hook:
        hook = layer.register_forward_hook(getActivation())
        hooks.append(hook)

    # Forward pass to capture activations
    out = model(x, type='feature_map')

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return activations


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


def proto(datas, loader, model):

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
                for j in range(args.num_concepts):
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
        
    def forward(self,labels,embeddings, means_y, means_n,mean_embedding_y, mean_embedding_n):
        batch_size = embeddings.shape[0]
        embedding_size = means_y.shape[1]
        num_parts = 2
        
        mean_embedding_y = mean_embedding_y.reshape(1, -1)
        mean_embedding_n = mean_embedding_n.reshape(1, -1)
        
        distances_y = torch.cdist(embeddings, mean_embedding_y)
        distances_n = torch.cdist(embeddings, mean_embedding_n)
        
        distances_im = torch.stack([distances_y, distances_n], dim=1)
        
        min_distances, val = torch.min(distances_im, dim=1)
        val=val.flatten() #.to(device)
        
        part_embeddings = embeddings.reshape(batch_size, -1, embedding_size)
        #---------------------------
        means_all = torch.stack([means_y, means_n], dim = 0)  
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


# Define a recursive function to flatten the layers

class ResNet18EmbeddingRef(nn.Module):
    def __init__(self):
        super(ResNet18EmbeddingRef, self).__init__()
        resnet = resnet18(pretrained=True)
        
        # Extracting the layers up to Layer 4
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # Residual layers
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4


    def forward(self, x):
        output = []
        
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # First ReLU
        x = self.maxpool(x)

        # Layer 1
        x = self.layer1(x)

        # Layer 2
        x = self.layer2(x)

        # Layer 3
        for block in self.layer3:
            residual = x
            # conv1 -> ReLU
            x = block.conv1(x)
            x = block.bn1(x)
            x = block.relu(x)  # Capture ReLU activation here
            output.append(x)  # Append ReLU output from Layer 3
            # conv2
            x = block.conv2(x)
            x = block.bn2(x)
            if hasattr(block, 'downsample') and block.downsample != None:
                residual = block.downsample(residual)
                x += residual

        # Layer 4
        for block in self.layer4:
            # conv1 -> ReLU
            residual = x
            x = block.conv1(x)
            x = block.bn1(x)
            x = block.relu(x)  # Capture ReLU activation here
            output.append(x)  # Append ReLU output from Layer 4
            # conv2
            x = block.conv2(x)
            x = block.bn2(x)
            if hasattr(block, 'downsample') and block.downsample != None:
                residual = block.downsample(residual)
                x += residual

        # Return the last 4 ReLU activations
        return output[-4:]
    

class ResNet18Embedding(nn.Module):
    def __init__(self):
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
        # Flatten the entire ResNet18 model
        # Average pooling and fully connected layer
        self.avgpool = resnet.avgpool

        # placeholder for the gradients
        self.gradients = None
        self.activation = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x, type=None):

        if type == "activation": 

            output = []
        
            for i,layer in enumerate(self.layer4): 
                if i<6 : 
                   x = layer(x)

                if(i==6): 
                   
                   # Layer 3
                   for target_layer, block in enumerate(layer):
                        residual = x
                        # conv1 -> ReLU
                        x = block.conv1(x)
                        x = block.bn1(x)
                        x = block.relu(x)  # Capture ReLU activation here
                        if(target_layer==1):
                            self.activation = x
                            h = x.register_hook(self.activations_hook)
                        output.append(x)  # Append ReLU output from Layer 3
                        # conv2
                        x = block.conv2(x)
                        x = block.bn2(x)
                        if hasattr(block, 'downsample') and block.downsample != None:
                            residual = block.downsample(residual)
                            x += residual


                elif(i==7):     

                    # Layer 4
                    for block in layer:
                        # conv1 -> ReLU
                        residual = x
                        x = block.conv1(x)
                        x = block.bn1(x)
                        x = block.relu(x)  # Capture ReLU activation here
                        output.append(x)  # Append ReLU output from Layer 4
                        # conv2
                        x = block.conv2(x)
                        x = block.bn2(x)
                        if hasattr(block, 'downsample') and block.downsample != None:
                            residual = block.downsample(residual)
                            x += residual


            
       

            # Return the last 4 ReLU activations
            return output[-4:]
            
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
class MseDirectionLoss(nn.Module):
    def __init__(self, lamda=0.5):
        super(MseDirectionLoss, self).__init__()
        self.lamda = lamda
        self.criterion = nn.MSELoss()
        self.similarity_loss = torch.nn.CosineSimilarity()

    def forward(self, output_pred, output_real, labels):

        mask= (labels == 0)  # Mask for positive samples
        y_pred_0, y_pred_1, y_pred_2, y_pred_3 = output_pred[0], output_pred[1], output_pred[2], output_pred[3]
        y_0, y_1, y_2, y_3 = output_real[0], output_real[1], output_real[2], output_real[3]
        #---------------------------- apply mask
        y_pred_0, y_pred_1, y_pred_2, y_pred_3 = y_pred_0[mask], y_pred_1[mask], y_pred_2[mask], y_pred_3[mask]
        y_0, y_1, y_2, y_3 = y_0[mask], y_1[mask], y_2[mask], y_3[mask]
        #---------------------------
        # different terms of loss
        abs_loss_0 = self.criterion(y_pred_0, y_0)
        loss_0 = torch.mean(1 - self.similarity_loss(y_pred_0.view(y_pred_0.shape[0], -1), y_0.view(y_0.shape[0], -1)))
        abs_loss_1 = self.criterion(y_pred_1, y_1)
        loss_1 = torch.mean(1 - self.similarity_loss(y_pred_1.view(y_pred_1.shape[0], -1), y_1.view(y_1.shape[0], -1)))
        abs_loss_2 = self.criterion(y_pred_2, y_2)
        loss_2 = torch.mean(1 - self.similarity_loss(y_pred_2.view(y_pred_2.shape[0], -1), y_2.view(y_2.shape[0], -1)))
        abs_loss_3 = self.criterion(y_pred_3, y_3)
        loss_3 = torch.mean(1 - self.similarity_loss(y_pred_3.view(y_pred_3.shape[0], -1), y_3.view(y_3.shape[0], -1)))

        total_loss = loss_0 + loss_1 + loss_2 + loss_3 + self.lamda * (
                abs_loss_0 + abs_loss_1 + abs_loss_2 + abs_loss_3)

        return total_loss


def train(args, tf_writer):

    # Load the pretrained model
    # model_pretrained = ResNet18Embedding_ref().cuda()
    
    model_pretrained = ResNet18EmbeddingRef().cuda()
    # for param in model_pretrained.parameters():
    #              param.requires_grad = False

    train_transform = transforms.Compose([
        transforms.Resize((args.im_size, args.im_size)),
        transforms.RandomHorizontalFlip(0.2),
        transforms.RandomVerticalFlip(0.1),
        transforms.RandomAutocontrast(0.2),
        transforms.ToTensor()
    ])
    train_dataset = BrainDataset('./train_image_paths.txt', './centroidss.txt', transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
  
    mod=ResNet18Embedding()
    # num_ftrs = mod.fc.in_features
    # mod.fc = nn.Linear(num_ftrs, args.num_concepts) #args.num_classes
    mod.to(device)
    #--------------------
    criterion1 = MeanDistanceLoss()
    criterion2 = MseDirectionLoss()
    #--------------------
    optimizer = optim.Adam(mod.parameters(), lr=0.001)
    max_acc=0
    max_acc_mod=0
    
    for epoch in range(40):
        print("Training Epoch", epoch)
        running_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        correct=0
        
        for batch in train_dataloader:
            # centroids = batch['centroids']
            labels = batch['labels'].to(device)
            
            # embeddings = parse_feature(batch, mod, args.num_concepts)
    
            # print("Computing Prototypes(Training)...")
            # mean_embedding_y, mean_embedding_n, means_y, means_n = proto(11, train_dataloader, mod)
    
            # print("Computing Concept Prototypes(Training)...")
            
            # distances, val, loss = criterion1(labels,embeddings, means_y, means_n, mean_embedding_y, mean_embedding_n)            
            # loss_p = F.cross_entropy(-distances.to(device), labels.unsqueeze(-1))
            #-----------------------------------------------
            output_real = model_pretrained(batch['image'].to(device))
            output_pred = mod(batch['image'].to(device), type= "activation")
            loss_salehi = criterion2(output_pred, output_real, labels)
            #-----------------------------------------------
            losstt = loss_salehi
            #-----------------------------------------------
            print("Loss: ", losstt.item())
            
            optimizer.zero_grad()
            losstt.backward()
            optimizer.step()
            #--------------
            running_loss += losstt.item()
            #--------------
            # batch_correct = torch.eq(val.to(device), labels).sum().item() #.to('cpu')
            # epoch_correct += batch_correct
            # epoch_total += len(labels)
        
        if(epoch == 39) : 
            torch.save({'epoch':epoch, 'state':mod.state_dict()}, 'resnet18_model_brain_v2_39epoch_27sep_salehi.pth')

        # accuracy_model = 100 * epoch_correct / epoch_total
        # tf_writer.add_scalar('loss/train', running_loss, epoch)
        # tf_writer.add_scalar('acc/train', accuracy_model, epoch)
        
        # print('Epoch {}: Accuracy(train set) = {:.2f}%'.format(epoch, accuracy_model))
  
        #-----------------------------------
        #-----------------------------------

        # val_dataset = BrainDataset('./test_image_paths.txt', './centroids_test.txt', transform=train_transform)
        # val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4)
        # epoch_total_val = 0
        # epoch_correct_val=0
        # for batch in val_dataloader:
        #     images = batch['image'].to(device)
        #     # centroids = batch['centroids']
        #     labels = batch['labels'].to(device)
            
        #     with torch.no_grad():
        #         embeddings = parse_feature(batch, mod, args.num_concepts)

            
        #     print("Computing Prototypes(val)...")
        #     mean_embedding_y, mean_embedding_n,_,_ = proto(11, val_dataloader, mod)
        #     #-----------------------------------
        #     distances_y = torch.cdist(embeddings, mean_embedding_y.reshape(1, -1))
        #     distances_n = torch.cdist(embeddings, mean_embedding_n.reshape(1, -1))
        #     #-----------------------------------
        #     distances_im = torch.stack([distances_y, distances_n], dim=1)
        #     #-----------------------------------
        #     min_distances, val = torch.min(distances_im, dim=1)
        #     val = val.view(-1)
        #     batch_correct_val= torch.eq(val.to(device), labels).sum().item() #.to('cpu')
        #     epoch_correct_val += batch_correct_val
        #     epoch_total_val += len(labels)
            
        # #---------------------------
        # accuracy_model_val = 100 * epoch_correct_val / epoch_total_val   
        # tf_writer.add_scalar('acc/test', accuracy_model_val, epoch)


        # if accuracy_model_val > max_acc : 
        #     max_acc = accuracy_model_val
        #     outfile = 'best_model_v2_brain_val_20epoch_25sep_salehi.tar'
        #     torch.save({'epoch':epoch, 'state':mod.state_dict()}, outfile)
        # if accuracy_model > max_acc_mod : 
        #     print("best model for model! save...")
        #     max_acc_mod = accuracy_model
        #     outfile = 'best_model_v2_brain_train_20epoch_25sep_salehi.tar'
        #     torch.save({'epoch':epoch, 'state':mod.state_dict()}, outfile)

        # if (epoch % 5==0) or (epoch==10-1):
        #     # outfile = ('{:d}.tar'.format(epoch))
        #     torch.save({'epoch':epoch, 'state':mod.state_dict()}, 'resnet18_model_brain_v2_20epoch_25sep_salehi.pth')
            
        # print('Epoch {}: Accuracy(test set) = {:.2f}%'.format(epoch, accuracy_model_val))


if __name__ == '__main__':

    args = parse_args()
     
    print("Training Started...")
    tf_writer = SummaryWriter(log_dir="./runs/run1_debug_25sep_v2_brain_20epoch_salehi")
    train(args, tf_writer)
    # train(mean_embedding_y, mean_embedding_n)
    