import torch
from torch import nn
from torchvision.models import resnet18

device = 'cuda'

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
