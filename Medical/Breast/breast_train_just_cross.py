import torch
import torch.nn as nn
import torch.optim as optim
#from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import torchvision.transforms as transforms
from PIL import Image
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


def proto(datas, model1):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(0.2),
        transforms.RandomVerticalFlip(0.1),
        transforms.RandomAutocontrast(0.2),
        transforms.ToTensor()
    ])
    #model1=ResNet18Embedding(64)
    #num_ftrs = model1.fc.in_features
    #model1.fc = nn.Linear(num_ftrs, 64)
    #model1.to(device)
    
    if datas==11:
      with open('./train_path.txt', 'r') as f:
          data = f.readlines()
    elif datas==22:
      with open('./vals_path.txt', 'r') as f:
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
            
    embeddings_b = torch.zeros(64).to(device)
    count_b = 0
    embeddings_m = torch.zeros(64).to(device)
    count_m = 0
    embeddings_n = torch.zeros(64).to(device)
    count_n = 0
    
    for i, path in enumerate(image_paths):
        image = Image.open(path).convert('RGB')
        image_tensor = transform(image).to(device)
        
        embedding = model1(image_tensor.unsqueeze(0), 'embedding').squeeze().detach()
        
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




class MeanDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,labels,embeddings, feature_maps, means_b, means_m,means_n, centroids,mean_embedding_b,mean_embedding_m,mean_embedding_n):
        batch_size = embeddings.shape[0]
        embedding_size = feature_maps.shape[1]
        num_parts = 64
        
        mean_embedding_b = mean_embedding_b.reshape(1, -1)
        mean_embedding_m = mean_embedding_m.reshape(1, -1)
        mean_embedding_n = mean_embedding_n.reshape(1, -1)
        
        distances_b = torch.cdist(embeddings, mean_embedding_b)
        distances_m = torch.cdist(embeddings, mean_embedding_m)
        distances_n = torch.cdist(embeddings, mean_embedding_n)
        
        distances_im = torch.stack([distances_b, distances_m, distances_n], dim=1)
        
        min_distances, val = torch.min(distances_im, dim=1)
        val=val.flatten() #.to(device)
        
        part_embeddings = torch.zeros(batch_size, num_parts, embedding_size).to(device)
        
        for i in range(batch_size):
            xs = [int(t[i]) for t in centroids['x']]
            ys = [int(t[i]) for t in centroids['y']]
            
            for j in range(num_parts):
                x = xs[j]
                y = ys[j]
                #------------------------
                x, y = int(x/28), int(y/28)
                embedding = feature_maps [i, :, x, y]
                # if x < 20 or y < 20:
                #     crop = images[i, :, 0:(x+40), 0:(y+40)]
                # else:
                #     crop = images[i, :, (x-20):(x+20), (y-20):(y+20)]
                    
                # with torch.no_grad():
                #     part_embedding = model(crop.unsqueeze(0))
                    
                part_embeddings[i, j, :] = embedding
                
        ## distances_b = torch.cdist(part_embeddings, means_b.unsqueeze(0))
        ## distances_m = torch.cdist(part_embeddings, means_m.unsqueeze(0))
        ## distances_n = torch.cdist(part_embeddings, means_n.unsqueeze(0))
        #---------------------------
        means_all = torch.stack([means_b, means_m, means_n], dim = 0)  
        dis = torch.norm(part_embeddings.unsqueeze(1) - means_all.unsqueeze(0), dim=-1)
        #---------------------------
        # d = torch.stack([distances_b, distances_m, distances_n], dim=1)
        closest_class = torch.argmin(dis, dim=1)
        same_class_mask = labels.view(-1, 1).to("cpu") == torch.arange(dis.shape[1])
        same_class_mask = same_class_mask.unsqueeze(2).expand(-1, -1, num_parts)
        #------------------
        different_class_mask = ~ same_class_mask
        #------------------
        #batch_size = part_embeddings.size(0)
        #num_parts = part_embeddings.size(1)
        
        # mask = torch.eye(num_parts).repeat(batch_size, 1, 1).to(device)
        # # distances = torch.cdist(part_embeddings, part_embeddings)
        # same_class_distances = dis * mask
        # different_class_distances = dis * (1 - mask)
        # same_class_distances = same_class_distances.to(device)
        # different_class_distances = different_class_distances.to(device)
        
        # mean_b_same = torch.sum(same_class_distances[closest_class == 0]) / torch.sum(closest_class == 0)
        # mean_b_different = torch.sum(different_class_distances[closest_class == 0]) / torch.sum(closest_class == 0)
        
        # mean_m_same = torch.sum(same_class_distances[closest_class == 1]) / torch.sum(closest_class == 1)
        # mean_m_different = torch.sum(different_class_distances[closest_class == 1]) / torch.sum(closest_class == 1)
        
        # mean_n_same = torch.sum(same_class_distances[closest_class == 2]) / torch.sum(closest_class == 2)
        # mean_n_different = torch.sum(different_class_distances[closest_class == 2]) / torch.sum(closest_class == 2)
        
        # margin = 1
        
        # loss_same = torch.max(torch.zeros(1).to(device), mean_b_same + mean_m_same+mean_n_same - 2 * margin)
        # loss_different = torch.max(torch.zeros(1).to(device), 2 * margin - mean_b_different - mean_m_different - mean_n_different)
        # loss = loss_same + loss_different
         
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
    means_b = [None] * num_concepts
    means_m = [None] * num_concepts
    means_n = [None] * num_concepts
    
    count_b = 0
    count_m = 0
    count_n = 0
    
    for batch in train_compute:

        feature_maps = model(batch['image'].to(device), 'feature_map')
        # images= batch['image'].to(device)
        
        for i in range(feature_maps.shape[0]):
            # image_i = images[i]
            # image_id = int(batch['image_id'][i])
            centroids = batch['centroids']
            labels=batch['labels'][i]
            
            part_ids = [int(t[i]) for t in centroids['part_ids']]
            xs=[int(t[i]) for t in centroids['x']]
            ys=[int(t[i]) for t in centroids['y']]
            
            for part_id, x, y in zip(part_ids,xs,ys):
                x, y = int(x/28), int(y/28)
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
                        
    for i in range(num_concepts):
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
        self.fc = nn.Linear(resnet.fc.in_features, embedding_size)
    
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
           x = self.fc(x)
           return x 
        
        else : 
            x = self.avgpool(x_prime)
            # Flatten and pass through the fully connected layer
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x, x_prime
        # Return both the embedding and the feature maps at Layer 4

def train(args, tf_writer):
    train_dataloader, val_dataloader = get_breast_data(args.im_size, args.batch_size)
    mod=ResNet18Embedding(args.num_classes)
    num_ftrs = mod.fc.in_features
    mod.fc = nn.Linear(num_ftrs, args.num_classes) #args.num_classes
    mod.to(device)
    #--------------------
    criterion = nn.CrossEntropyLoss()
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
            images = batch['image'].to(device)
            centroids = batch['centroids']
            labels = batch['labels'].to(device)
            
            embeddings, feature_maps = mod(images)
            
            #print('emb size', embeddings.shape, labels.shape)
            # loss_g = F.cross_entropy(embeddings, labels) # saeed : why????!
            
            # print("Computing Prototypes(Training)...")
            # mean_embedding_b, mean_embedding_m, mean_embedding_n = proto(11, mod)
    
            # print("Computing Concept Prototypes(Training)...")
            # means_b, means_m, means_n = compute_means(train_dataloader, mod, args.num_concepts)
            #mean_embedding_b, mean_embedding_m, mean_embedding_n=Variable(mean_embedding_b), Variable(mean_embedding_m), Variable(mean_embedding_n)
            
            # distances, val, loss = criterion(labels,embeddings, feature_maps, means_b,means_m, means_n, centroids, mean_embedding_b, mean_embedding_m,mean_embedding_n)            
            
            #print(distances.min(), distances.max(), embeddings.min(), embeddings.max(), labels.min(), labels.max())
            #print(distances.shape)
            # loss_p = F.cross_entropy(-distances, labels.unsqueeze(-1))
            
            # losstt = loss +  loss_p 
            losstt = criterion(embeddings , labels)
            
            print("Loss: ", losstt.item())
            
            optimizer.zero_grad()
            losstt.backward()
            optimizer.step()
            
            running_loss += losstt.item()
            
            _ , val = torch.max(embeddings, dim=1)
            val = val.view(-1)
            #labels=labels.to('cpu')
            batch_correct = torch.eq(val, labels).sum().item() #.to('cpu')
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
            centroids = batch['centroids']
            labels = batch['labels'].to(device)
            
            with torch.no_grad():
                embeddings = mod(images, 'embedding') #.to(device) 
            
            
            # print("Computing Prototypes(val)...")
            # mean_embedding_b, mean_embedding_m, mean_embedding_n = proto(22, mod)
            # #-----------------------------------
            # distances_b = torch.cdist(embeddings, mean_embedding_b.reshape(1, -1))
            # distances_m = torch.cdist(embeddings, mean_embedding_m.reshape(1, -1))
            # distances_n = torch.cdist(embeddings, mean_embedding_n.reshape(1, -1))
            # #-----------------------------------
            # distances_im = torch.stack([distances_b, distances_m, distances_n], dim=1)
            #-----------------------------------
            max_distances, val = torch.max(embeddings, dim=1)
            val = val.view(-1)
            batch_correct_val= torch.eq(val, labels).sum().item() #.to('cpu')
            epoch_correct_val += batch_correct_val
            epoch_total_val += len(labels)
            
        #---------------------------
        accuracy_model_val = 100 * epoch_correct_val / epoch_total_val   
        tf_writer.add_scalar('acc/test', accuracy_model_val, epoch)


        # if accuracy_model_val > max_acc : 
        #     max_acc = accuracy_model_val
        #     outfile = 'best_model_model.tar'
        #     torch.save({'epoch':epoch, 'state':mod.state_dict()}, outfile)
        # if accuracy_model > max_acc_mod : 
        #     print("best model for model! save...")
        #     max_acc_mod = accuracy_model
        #     outfile = 'best_model_model.tar'
        #     torch.save({'epoch':epoch, 'state':mod.state_dict()}, outfile)

        # if (epoch % 5==0) or (epoch==10-1):
        #     # outfile = ('{:d}.tar'.format(epoch))
        #     torch.save({'epoch':epoch, 'state':mod.state_dict()}, 'resnet18_model.pth')
            
        print('Epoch {}: Accuracy(test set) = {:.2f}%'.format(epoch, accuracy_model_val))


if __name__ == '__main__':
    args = parse_args()
     
    print("Training Started...")
    tf_writer = SummaryWriter(log_dir="./runs/just_simple_cross_entropy_loss")
    train(args, tf_writer)