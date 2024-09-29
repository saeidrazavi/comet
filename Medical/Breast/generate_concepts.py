import os
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
#set paths
benign_path = './breast-ultrasound-images-dataset/benign/'
malig_path = './breast-ultrasound-images-dataset/malignant/'
norm_path='./breast-ultrasound-images-dataset/normal/'
image_paths = []
j=0
for i, filename in enumerate(os.listdir(benign_path)):
    if "_mask" not in filename:
        image_paths.append((j+1, os.path.join(benign_path, filename)))
        j=j+1
k=0
for i, filename in enumerate(os.listdir(malig_path)):
    if "_mask" not in filename:
        image_paths.append((k+437+1, os.path.join(malig_path, filename)))
        k=k+1
l=0
for i, filename in enumerate(os.listdir(norm_path)):
    if "_mask" not in filename:
        image_paths.append((l+647+1, os.path.join(norm_path, filename)))
        l=l+1
data, test_data = train_test_split(image_paths, test_size=0.2, random_state=1357)
train_data, val_data = train_test_split(data, test_size=0.1, random_state=2468)

with open('train_path.txt', 'w') as f:
    for image_id, image_path in train_data:
        f.write(str(image_id) + ' ' + image_path + '\n')
with open('test_path.txt', 'w') as f:
    for image_id, image_path in test_data:
        f.write(str(image_id) + ' ' + image_path + '\n')
with open('vals_path.txt', 'w') as f:
    for image_id, image_path in val_data:
        f.write(str(image_id) + ' ' + image_path + '\n')

class Grid():
    def __init__(self, paths, augmentations=None, grid_size=(8, 8)):
        self.paths = paths
        self.grid_size = grid_size
        
        if augmentations is None:
            self.augmentations = transforms.Compose([transforms.ToTensor()])
        else:
            self.augmentations = augmentations
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
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
        with open('centroid.txt', 'a') as f:
            for i in range(gs_y):
                for j in range(gs_x):
                    x = j * w_step + w_step // 2
                    y = i * h_step + h_step // 2
                    part = f'{i*gs_x+j+1}'
                    centroids.append((part, x, y))
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0, 255, 0))
                    f.write(f'{index+1} {part} {x} {y}\n')
        return (sample)

train_augmentations = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

trainer=[]
tester=[]
valer=[]
with open('train_path.txt', 'r') as f:
    for line in f:
        img_id, img_path = line.strip().split(" ",1)
        trainer.append(img_path)
with open('test_path.txt', 'r') as f:
    for line in f:
        img_id, img_path = line.strip().split(" ",1)
        tester.append(img_path)
with open('vals_path.txt', 'r') as f:
    for line in f:
        img_id, img_path = line.strip().split(" ",1)
        valer.append(img_path)

dataset = Grid(trainer+tester+valer, augmentations=train_augmentations, grid_size=(8, 8))
for i in range(len(dataset)):
 j = dataset[i]
example_item=dataset[44]

#Remove redundant entries
filename = 'centroid.txt'
num_entries_to_remove = 64
with open(filename, 'r') as f:
    lines = f.readlines()
f.close()

with open(filename, 'w') as f:
    for line in lines[:-num_entries_to_remove]:
        f.write(line)
f.close()


train_dict = {}
with open('train_path.txt', 'r') as f:
    for line in f:
        img_id, img_path = line.strip().split(" ",1)
        train_dict[int(img_id)] = img_path

centroids_train = []
with open('centroid.txt', 'r') as f:
    for line in f:
        img_id, part, x, y = map(int, line.strip().split())
        if img_id in train_dict:
            centroids_train.append((img_id, part, x, y))

with open('centroid_train.txt', 'w') as f:
   for centroid in centroids_train:
       f.write('{} {} {} {}\n'.format(*centroid)) #centroids

#-----------------------------
#-----------------------------

test_dict = {}
with open('test_path.txt', 'r') as f:
    for line in f:
        img_id, img_path = line.strip().split(" ",1)
        test_dict[int(img_id)] = img_path

centroids_test = []
with open('centroid.txt', 'r') as f:
    for line in f:
        img_id, part, x, y = map(int, line.strip().split())
        if img_id in test_dict:
            centroids_test.append((img_id, part, x, y))

with open('centroid_tester.txt', 'w') as f:
    for centroid in centroids_test:
        f.write('{} {} {} {}\n'.format(*centroid)) #centroids

#-------------------------------
#-------------------------------

val_dict = {}
with open('vals_path.txt', 'r') as f:
    for line in f:
        img_id, img_path = line.strip().split(" ",1)
        val_dict[int(img_id)] = img_path

centroids_val = []
with open('centroid.txt', 'r') as f:
    for line in f:
        img_id, part, x, y = map(int, line.strip().split())
        if img_id in val_dict:
            centroids_val.append((img_id, part, x, y))

with open('centroid_val.txt', 'w') as f:
    for centroid in centroids_val:
        f.write('{} {} {} {}\n'.format(*centroid)) #centroids
