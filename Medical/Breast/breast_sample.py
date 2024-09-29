import torch
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage import exposure,util
import numpy as np
from kmeans_pytorch import kmeans
from skimage.color import label2rgb
from skimage.color import gray2rgb
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
from skimage import io
from skimage import transform
from skimage.measure import regionprops
from breast_eval import compute_means
import torch.nn as nn
import torch.nn.functional as F
from data import get_breast_data
from torchvision.models import resnet18
import cv2
import requests
from PIL import Image
# from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam import GradCAMElementWise
# from pytorch_grad_cam import XGradCAM
# from pytorch_grad_cam import LayerCAM

class SimilarityToConceptTarget:
    def __init__(self, features):
        self.features = features
    
    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity(dim=0)
        return cos(model_output, self.features)

class DifferenceFromConceptTarget:
    def __init__(self, features):
        self.features = features
    
    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity(dim=0)
        return 1 - cos(model_output, self.features)

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
        self.avgpool = resnet.avgpool
        
    def forward(self, x, type=None):
        x_prime = self.layer4(x)
        if type == 'feature_map':
           
           return x_prime
        
        if type == 'embedding':
           
           x = self.avgpool(x_prime)
           x = x.view(x.size(0), -1)
           return x 
        
        else : 
            x = self.avgpool(x_prime)
            x = x.view(x.size(0), -1)
            return x, x_prime

def k_means_compute(embeddings, flag = None):
    
    device = 'cpu'
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
                
            feature_maps = model(batch['image'], 'feature_map')
            #----------------
            labels = batch['labels']
            total_labels.append(labels)
            #----------------
            # feature_avg = globalpool(feature_maps).view(feature_maps.size(0), feature_maps.size(1))
            batch_num = feature_maps.size(0)

            feat_list = []
            for i in range(batch_num):
                feat = []
                for j in range(64):
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
        
        embeddings_b, embeddings_m, embeddings_n= torch.cat(embeddings_b,dim=0), torch.cat(embeddings_m,dim=0), torch.cat(embeddings_n,dim=0)
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
        
def parse_feature(image, model, num_concepts):
    
        # globalpool = nn.AdaptiveAvgPool2d((1,1))
        feature_maps = model(image, 'feature_map')
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
            # feat = k_means_compute(feat, 'parse_feature')
            feat_list.append(feat.view(1, -1))

        z_all = torch.cat(feat_list, dim=0)
        #-----------
        return z_all


# # #for a random sample, get values
im="./benign (7).png" #set path to sample image
image = img_as_float(io.imread(im))
image = transform.resize(image, (256, 256))
#segments = slic(image, n_segments=70, compactness=20, sigma=2)
segments = slic(image, n_segments=75, compactness=50, sigma=1)
# props = regionprops(segments)

image = Image.open("./benign (7).png").convert('RGB')
# image = transform.resize(image, (256, 256))
train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
# image = torch.from_numpy(image.transpose((2, 0, 1))).float().div(255)
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
image = train_transform(image).unsqueeze(0)

# img = np.array(Image.open("./test2.png"))
# img = cv2.resize(img, (256, 256))
# input_float = np.float32(img) / 255

mod=ResNet18Embedding(64)
modi= torch.load('./best_model_model_v4_train.tar')
msg = mod.load_state_dict(modi['state'])
print(msg)

with torch.no_grad():
    part_embeddings = parse_feature(image, mod, 64)

part_embeddings=part_embeddings.to('cuda')
part_embeddings = part_embeddings.reshape(64, -1)

train_loader, val_loader = get_breast_data(256, 32)
print("Computing Prototypes(val)...")
mean_embedding_b , mean_embedding_m , mean_embedding_n , means_b, means_m, means_n = proto(11, train_loader, mod)
means_normal_b = means_b[1]

    
# target_layers = [mod.layer4[-1]]
# target_layers = [target_layers[-1][-1]]
# # car_targets = [SimilarityToConceptTarget(car_concept_features)]
# embedding_n_target = [DifferenceFromConceptTarget(mean_embedding_n)]

# with GradCAMElementWise(model=mod,
#              target_layers=target_layers) as cam:
#     m_grayscale_cam = cam(input_tensor=image,
#                         targets=embedding_n_target)[0, :]

# benign_cam_image = show_cam_on_image(input_float, m_grayscale_cam, use_rgb=True)
# gradcamim= Image.fromarray(benign_cam_image)  
# gradcamim.save('GradCAMElementWise_test2_V3.png')                      
# # cloud_cam_image = show_cam_on_image(image_float, cloud_grayscale_cam, use_rgb=True)
# Image.fromarray(cloud_cam_image)
means_normal_b=means_normal_b.to('cuda')
dist_take=torch.cdist(part_embeddings, means_normal_b.unsqueeze(0), p=3) #p=2
# dist_take = torch.norm(part_embeddings - means_b, dim =1)
#print("dt: ", dist_take.shape, part_embeddings.shape, means_m.shape)
weights=dist_take.tolist()
weights_np = np.array(weights).flatten()
weights_np = (weights_np - np.min(weights_np)) / (np.max(weights_np) - np.min(weights_np)) #

# print(weights_np.shape)
# weights_np = weights_np.mean(2).flatten()
# print(weights_np.shape)
# weights_np = weights_np.flatten()
# np.savetxt('weight_np.txt', weights_np)


# # sample weights_np
# weights_np_2 =[0.38057508169993065, 0.7211149805850931, 0.2524438469404538, 0.26513683397572957,
#             0.14942684906429446, 0.6441100964667015, 0.612196709119643, 0.6040145593924529, 
#             0.5160074656216703, 0.7176650624637981, 0.6231791820710664, 0.589004655287076,
#             0.6321965661000708, 0.7448745360802626, 0.5919937643466509, 0.39045773414091717,
#             0.820890868915411, 0.6359579808496792, 0.5828977195529208, 0.5382970659534757,
#             0.6649909540120565, 0.6502813910083595, 0.5118956529201021, 0.6312154518345836,
#             0.6791355897054512, 0.8376672792671677, 0.9726298438941369, 0.36532919530037683,
#             0.6537067097632311, 0.7224015846568603, 0.0, 0.6346278988279546, 0.7293926673865319,
#             0.3872540957229997, 0.6341559342396007, 0.613039308929427, 0.5833482312054404, 
#             0.5981650588883088, 0.70, 0.6701253566550582, 0.6135317968263957, 0.5694753326992799,
#             0.612588582747549, 0.6443746826753242, 0.6502170322008567, 0.5907065881965947, 
#             0.6502170322008567, 0.6526755386474639, 0.6842120694217004, 0.6397766034281791, 
#             0.576054233021789, 0.6079833525217926, 0.55651060847677, 0.5244663582211225, 
#             0.6125170729614348, 0.5093427535558241, 0.5417931793966004, 0.6189386517544926,
#             0.6389184859948084, 0.5451405524846075, 0.509663832495477]
# ''' '''


image = img_as_float(io.imread(im))
image = transform.resize(image, (256, 256))
# mask = Image.open("./test2_mask.png").convert("RGB")
# mask_o = mask.resize((256, 256))
# #segments = slic(image, n_segments=70, compactness=35, sigma=1)
# segments = slic(image, n_segments=70, compactness=20, sigma=1)

fig = plt.figure("Image")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(image,cmap='binary')
plt.axis("off")
#plt.show()
plt.imsave('im1.png', image)
#mask_o.show()
#plt.show()

cmap = plt.cm.get_cmap('Reds')
colors = cmap(weights_np)
colored_segments = label2rgb(segments, image=image, colors=colors, kind='overlay', alpha=0.8)

plt.imshow(colored_segments)
plt.axis('off')
plt.imsave('im2.png', colored_segments)

a=[]
threshold = 0.005

for i in weights_np:
    if i<threshold:
        a.append(float(0))
    else:
        a.append(float(1))
    
binary_weights = a

cmap = plt.cm.get_cmap('binary')
masks = cmap(binary_weights)
masked_segments = label2rgb(segments, colors=masks, bg_label=-1)

plt.imshow(masked_segments)
plt.axis('off')
#plt.show()
plt.imsave('im3.png', masked_segments)