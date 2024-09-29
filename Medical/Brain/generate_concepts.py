import os
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import os
from sklearn.model_selection import train_test_split
dataset_path = './brain' #set path to dataset
paths = []
labels = []
for label in ['yes','no']:
    for dirname, _, filenames in os.walk(os.path.join(dataset_path,label)):
        for filename in filenames:
            paths.append(os.path.join(dirname, filename))
            labels.append(1 if label is 'yes' else 0)
X_train, X_test, y_train, y_test = train_test_split(paths, labels, stratify=labels, test_size=0.2, shuffle=True)
test_augmentations = transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(0.2),transforms.ToTensor()])
class Grid():
    def __init__(self, paths, labels, augmentations=None, grid_size=(8,8)):
        self.paths = paths
        self.labels = labels
        self.grid_size = grid_size
        
        if augmentations is None:
            self.augmentations = transforms.Compose([transforms.ToTensor()])
        else:
            self.augmentations = augmentations
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        label = self.labels[index]
        sample = Image.open(self.paths[index]).convert(mode="RGB")
        sample = self.augmentations(sample)
        vis_image = transforms.ToPILImage()(sample).copy()
        w, h = vis_image.size
        gs_y, gs_x = self.grid_size
        h_step, w_step = h // gs_y, w // gs_x
        draw = ImageDraw.Draw(vis_image)
        for i in range(1, gs_x):
            x = i * w_step
            line = ((x, 0), (x, h))
            draw.line(line, fill=(255, 0, 0), width=2)

        for i in range(1, gs_y):
            y = i * h_step
            line = ((0, y), (w, y))
            draw.line(line, fill=(255, 0, 0), width=2)
        centroids = []
        with open('centroidss.txt', 'a') as f:
            for i in range(gs_y):
                for j in range(gs_x):
                    x = j * w_step + w_step // 2
                    y = i * h_step + h_step // 2
                    part = f'{i*gs_x+j+1}'
                    centroids.append((part, x, y))
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0, 255, 0))
                    f.write(f'{index+1} {part} {x} {y}\n')        
        return (sample, torch.tensor(label, dtype=torch.float), vis_image)
data = Grid(X_test+X_train, y_test+y_train, augmentations=test_augmentations, grid_size=(8, 8))
for i in range(len(data)):
 j = data[i]
example_item=data[44]

#Remove redundant entries
filename = 'centroidss.txt'
num_entries_to_remove = 64
with open(filename, 'r') as f:
    lines = f.readlines()
f.close()
with open(filename, 'w') as f:
    for line in lines[:-num_entries_to_remove]:
        f.write(line)
f.close()
#set path to data
yes_path = './brain/yes/' 
no_path = './brain/no/' 
image_paths = []
for i, filename in enumerate(os.listdir(yes_path)):
    image_paths.append((i+1, os.path.join('yes', filename)))
for i, filename in enumerate(os.listdir(no_path)):
    image_paths.append((i+len(os.listdir(yes_path))+1, os.path.join('no', filename)))
train_paths, test_paths = train_test_split(image_paths, test_size=0.2, random_state=1357)
with open('train_image_paths.txt', 'w') as f:
    for image_id, image_path in train_paths:
        f.write(str(image_id) + ' ' + image_path + '\n')
with open('test_image_paths.txt', 'w') as f:
    for image_id, image_path in test_paths:
        f.write(str(image_id) + ' ' + image_path + '\n')
test_dict = {}
with open('test_image_paths.txt', 'r') as f:
    for line in f:
        img_id, img_path = line.strip().split(" ",1)
        test_dict[int(img_id)] = img_path
centroids_test = []
with open('centroidss.txt', 'r') as f:
    for line in f:
        img_id, part, x, y = map(int, line.strip().split())
        if img_id in test_dict:
            centroids_test.append((img_id, part, x, y))
with open('centroids_test.txt', 'w') as f:
    for centroid in centroids_test:
        f.write('{} {} {} {}\n'.format(*centroid))