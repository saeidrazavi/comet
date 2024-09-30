import torch
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# from skimage.segmentation import slic
# from skimage import exposure,util
import numpy as np
from torch.utils.data import Dataset, DataLoader
from kmeans_pytorch import kmeans
from skimage.color import label2rgb
from skimage.color import gray2rgb
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
from skimage import io
from skimage import transform
from torch.autograd import Variable
from brain_eval import BrainDataset
# from skimage.measure import regionprops
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
# import cv2
import requests
from PIL import Image
# from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam import GradCAMElementWise
# from pytorch_grad_cam import XGradCAM
# from pytorch_grad_cam import LayerCAM
device = 'cuda'

def grad_calc(input, model1, model2):

    input = input.cuda()
    input.requires_grad = True
    temp = torch.zeros(input.shape)
    lamda = 0.5
    criterion = nn.MSELoss()
    similarity_loss = torch.nn.CosineSimilarity()
    #---------------------------------------   
    output_pred = model1.forward(input, type="activation")
    output_real = model2(input)
    #---------------------------------------
    y_pred_1, y_pred_2, y_pred_3 = output_pred[1], output_pred[2], output_pred[3]
    y_1, y_2, y_3 = output_real[1], output_real[2], output_real[3]
    #---------------------------------------
    abs_loss_1 = criterion(y_pred_1, y_1)
    loss_1 = torch.mean(1 - similarity_loss(y_pred_1.view(y_pred_1.shape[0], -1), y_1.view(y_1.shape[0], -1)))
    abs_loss_2 = criterion(y_pred_2, y_2)
    loss_2 = torch.mean(1 - similarity_loss(y_pred_2.view(y_pred_2.shape[0], -1), y_2.view(y_2.shape[0], -1)))
    abs_loss_3 = criterion(y_pred_3, y_3)
    loss_3 = torch.mean(1 - similarity_loss(y_pred_3.view(y_pred_3.shape[0], -1), y_3.view(y_3.shape[0], -1)))
    total_loss = loss_1 + loss_2 + loss_3 + lamda * (abs_loss_1 + abs_loss_2 + abs_loss_3)
    #--------------------------------------
    model1.zero_grad()
    total_loss.backward()

    temp[0] = input.grad[0]

    return temp

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


# def proto(datas, loader, model):

#     # globalpool = nn.AdaptiveAvgPool2d((1,1))
#     num_concepts = 64
#     embeddings_y = []
#     embeddings_n = []
#     total_labels = []
    
#     if datas == 11:  
        
#         for batch in loader: 
                
#             feature_maps = model(batch['image'].to(device), 'feature_map')
#             #----------------
#             labels = batch['labels']
#             total_labels.append(labels)
#             #----------------
#             # feature_avg = globalpool(feature_maps).view(feature_maps.size(0), feature_maps.size(1))
#             batch_num = feature_maps.size(0)

#             feat_list = []
#             for i in range(batch_num):
#                 feat = []
#                 for j in range(num_concepts):
#                     x,y = int(j%8), int(j/8)
#                     feat.append(feature_maps[i, :, x, y])
#                 # feat.append(feature_avg[i, :])
#                 feat = torch.cat(feat, dim=0)
#                 feat_list.append(feat.view(1, -1))
#             z_all = torch.cat(feat_list, dim=0)
#             #---------------------
#             embeddings_y.append(z_all)
#             embeddings_n.append(z_all)
        
#         embeddings_y, embeddings_n= torch.cat(embeddings_y,dim=0).to(device), torch.cat(embeddings_n,dim=0).to(device)
#         labels_tensor = torch.cat(total_labels, dim=0).flatten()
#         #------------------------
#         mask_y = torch.eq(torch.zeros_like(labels_tensor), labels_tensor)
#         mask_n = torch.eq(torch.zeros_like(labels_tensor)+1, labels_tensor)
#         #-----------------------
#         embeddings_y = embeddings_y[mask_y].reshape(-1,512)
#         embeddings_n = embeddings_n[mask_n].reshape(-1,512)
#         #----------------------
#         # print("k-means b computing...")
#         centroids_y = k_means_compute(embeddings_y)
#         # print("k-means n computing...")
#         centroids_n = k_means_compute(embeddings_n)
#         #-----------------------
#         embeddings_y  = torch.cat((centroids_y[0],centroids_y[1]))
#         embeddings_n  = torch.cat((centroids_n[0],centroids_n[1]))


#         return embeddings_y, embeddings_n, embeddings_y.reshape(2,-1), embeddings_n.reshape(2,-1)


# def parse_feature(image, model, num_concepts):
    
#         # globalpool = nn.AdaptiveAvgPool2d((1,1))
#         feature_maps = model(image.cuda(), 'feature_map')
#         # feature_avg = globalpool(feature_maps).view(feature_maps.size(0), feature_maps.size(1))
#         batch_num = feature_maps.size(0)

#         feat_list = []
#         for i in range(batch_num):
#             feat = []
#             for j in range(num_concepts):
#                 x,y = int(j%8), int(j/8)
#                 feat.append(feature_maps[i, :, x, y])
#             # feat.append(feature_avg[i, :])
#             feat = torch.cat(feat, dim=0).reshape(64,-1)
#             # feat = k_means_compute(feat, 'parse_feature')
#             feat_list.append(feat.view(1, -1))

#         z_all = torch.cat(feat_list, dim=0)
#         #-----------
#         return z_all

# # #for a random sample, get values
im="./Y1.jpg" #set path to sample image
image = img_as_float(io.imread(im))
image = transform.resize(image, (256, 256))
#segments = slic(image, n_segments=70, compactness=20, sigma=2)
# segments = slic(image, n_segments=75, compactness=50, sigma=1)
# props = regionprops(segments)

image = Image.open(im).convert('RGB')
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

mod=ResNet18Embedding()
modi= torch.load('best_model_v2_brain_train_20epoch_25sep_salehi.tar')
msg = mod.load_state_dict(modi['state'])
mod = mod.cuda()
print(msg)
#----------------------
model_pretrained = ResNet18EmbeddingRef().cuda()

# with torch.no_grad():
#     part_embeddings = parse_feature(image, mod, 64)
heat_map = grad_calc(image, mod, model_pretrained)
# part_embeddings=part_embeddings.to('cuda')
# part_embeddings = part_embeddings.reshape(64, -1)

# train_dataset = BrainDataset('./train_image_paths.txt', './centroidss.txt', transform=train_transform)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
# #---------------------------  
# val_dataset = BrainDataset('./test_image_paths.txt', './centroids_test.txt', transform=train_transform)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4)

# print("Computing Prototypes(val)...")
# mean_embedding_y, mean_embedding_n , means_y, means_n = proto(11, train_loader, mod)
# means_normal_y = means_n[0]


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
# means_normal_y=means_normal_y.to('cuda')
# dist_take=torch.cdist(part_embeddings, means_normal_y.unsqueeze(0), p=3) #p=2
# dist_take = torch.norm(part_embeddings - means_b, dim =1)
#print("dt: ", dist_take.shape, part_embeddings.shape, means_m.shape)
# weights=dist_take.tolist()
# weights_np = np.array(weights).flatten()
# weights_np = (weights_np - np.min(weights_np)) / (np.max(weights_np) - np.min(weights_np)) #
# weights_np[weights_np>0.4] = 0
# weights_np /= weights_np.sum()
# print(weights_np.shape)
# weights_np = weights_np.mean(2).flatten()
# print(weights_np.shape)
# weights_np = weights_np.flatten()
# np.savetxt('weight_np.txt', weights_np)




# image = img_as_float(io.imread(im))
# image = transform.resize(image, (256, 256))
# mask = Image.open("./test2_mask.png").convert("RGB")
# mask_o = mask.resize((256, 256))
# #segments = slic(image, n_segments=70, compactness=35, sigma=1)
# segments = slic(image, n_segments=70, compactness=20, sigma=1)

# fig = plt.figure("Image")
# ax = fig.add_subplot(1, 1, 1)
# ax.imshow(image,cmap='binary')
# plt.axis("off")
# #plt.show()
# plt.imsave('im1.png', image)
#mask_o.show()
#plt.show()

# cmap = plt.cm.get_cmap('Reds')
# colors = cmap(weights_np)
# colored_segments = label2rgb(segments, image=image, colors=colors, kind='overlay', alpha=0.8)

# plt.imshow(colored_segments)
# plt.axis('off')
# plt.imsave('im2.png', colored_segments)

# a=[]
# threshold = 0.5

# for i in weights_np:
#     if i<threshold:
#         a.append(float(0))
#     else:
#         a.append(float(1))
    
# binary_weights = a

# cmap = plt.cm.get_cmap('binary')
# masks = cmap(binary_weights)
# masked_segments = label2rgb(segments, colors=masks, bg_label=-1)

# plt.imshow(masked_segments)
# plt.axis('off')
# #plt.show()
# plt.imsave('im3.png', masked_segments)

# Step 1: Create a random numpy array with 64 float numbers

# Step 2: Normalize the array so that the sum of the array equals 1

# Step 3: Reshape the array into an 8x8 grid
# grid = weights_np.reshape((8, 8))

# # Step 4: Load the image
# # Replace 'image_path' with your image file path
# img = Image.open(im).convert('L')  # Convert image to grayscale ('L' mode)

# # Step 5: Resize image to fit 8x8 grid (optional, for better visualization)
# img = img.resize((256, 256))  # Resize to 800x800 pixels

# # Step 6: Create a plot
# fig, ax = plt.subplots()

# # Show the grayscale image
# ax.imshow(img, cmap='gray')
# heatmap = ax.imshow(grid, cmap='viridis', alpha=0.6, extent=(0, 256, 256, 0))

# # Add a color bar to show the mapping of values
# plt.colorbar(heatmap, label='Normalized Value')
# ax.grid(True, color='white', linewidth=0.5)

# # Remove axis labels for a cleaner image
# ax.set_xticks([])
# ax.set_yticks([])

# # Step 7: Save the plot to an image file
# save_path = 'output_image.png'  # Specify your desired file path and format
# plt.savefig(save_path, bbox_inches='tight', dpi=300)  # dpi=300 for high quality
