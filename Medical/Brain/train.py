import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import argparse
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.stats import mode
from torch.utils.tensorboard import SummaryWriter
from utils import * 
from Losses.losses import *
from Networks.Resnets import *
from Networks.Vggs import * 

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
    parser.add_argument('--config', type=str, default='Config/config.yaml', help="training configuration")

    return parser.parse_args()

def train(args, tf_writer):

    # Load the pretrained model
    # model_pretrained = ResNet18Embedding_ref().cuda()
    config = get_config(args.config)
    #--------------------------------
    direction_loss_only = config["direction_loss_only"]
    normal_class = config["normal_class"]
    learning_rate = float(config['learning_rate'])
    num_epochs = config["num_epochs"]
    lamda = config['lamda']
    continue_train = config['continue_train']
    last_checkpoint = config['last_checkpoint']
    model_pretrained = ResNet18EmbeddingRef().cuda()
   
    #---------------------------------------
    train_transform = transforms.Compose([
        transforms.Resize((args.im_size, args.im_size)),
        transforms.RandomHorizontalFlip(0.2),
        transforms.RandomVerticalFlip(0.1),
        transforms.RandomAutocontrast(0.2),
        transforms.ToTensor()
    ])
    train_dataset = BrainDataset('./train_image_paths.txt', './centroidss.txt', transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    #---------------------------------------
    # mod=ResNet18Embedding()
    # num_ftrs = mod.fc.in_features
    # mod.fc = nn.Linear(num_ftrs, args.num_concepts) #args.num_classes
    # mod.to(device)
    vgg, model = get_networks(config)
    vgg, model = vgg.to(device), model.to(device)
    #--------------------
    criterion1 = MeanDistanceLoss()
    criterion2 = MseDirectionLoss(lamda)
    #--------------------
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    max_acc=0
    max_acc_mod=0
    
    for epoch in range(num_epochs):
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
            output_real = vgg(batch['image'].to(device))
            output_pred = model(batch['image'].to(device))
            #----------------------------------------------
            healthy_mask = (labels == 1)  # Mask for healthy samples
            output_pred = [output_pred[i][healthy_mask] for i in range(len(output_pred))]
            output_real = [output_real[i][healthy_mask] for i in range(len(output_real))]
            loss_salehi = criterion2(output_pred, output_real)
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
    
