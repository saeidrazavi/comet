import torch
from torch import nn

device = 'cuda'

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
        
        return distances_im, val, loss


class MseDirectionLoss(nn.Module):
    def __init__(self, lamda):
        super(MseDirectionLoss, self).__init__()
        self.lamda = lamda
        self.criterion = nn.MSELoss()
        self.similarity_loss = torch.nn.CosineSimilarity()

    def forward(self, output_pred, output_real):
        y_pred_0, y_pred_1, y_pred_2, y_pred_3 = output_pred[3], output_pred[6], output_pred[9], output_pred[12]
        y_0, y_1, y_2, y_3 = output_real[3], output_real[6], output_real[9], output_real[12]

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


class DirectionOnlyLoss(nn.Module):
    def __init__(self):
        super(DirectionOnlyLoss, self).__init__()
        self.similarity_loss = torch.nn.CosineSimilarity()

    def forward(self, output_pred, output_real):
        y_pred_0, y_pred_1, y_pred_2, y_pred_3 = output_pred[3], output_pred[6], output_pred[9], output_pred[12]
        y_0, y_1, y_2, y_3 = output_real[3], output_real[6], output_real[9], output_real[12]

        loss_0 = torch.mean(1 - self.similarity_loss(y_pred_0.view(y_pred_0.shape[0], -1), y_0.view(y_0.shape[0], -1)))
        loss_1 = torch.mean(1 - self.similarity_loss(y_pred_1.view(y_pred_1.shape[0], -1), y_1.view(y_1.shape[0], -1)))
        loss_2 = torch.mean(1 - self.similarity_loss(y_pred_2.view(y_pred_2.shape[0], -1), y_2.view(y_2.shape[0], -1)))
        loss_3 = torch.mean(1 - self.similarity_loss(y_pred_3.view(y_pred_3.shape[0], -1), y_3.view(y_3.shape[0], -1)))

        total_loss = loss_0 + loss_1 + loss_2 + loss_3

        return total_loss
