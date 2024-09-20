
!pip install google-streetview

from google.colab import drive
drive.mount('/content/drive')

import google_streetview.api

result_path='/content/drive/MyDrive/Road Crack Detection/'

# Define a list of locations
a,b = 41.6016424,-87.5093424
locations = [
    {'location': '{}, {}'.format(a, b)},
    # Add more locations here
]

for i in range(0,7500):
    d ={'location': '{}, {}'.format(a, b)}
    locations.append(d)
    a = a + 0.0010
    b = b + 0.0030

# Define parameters for each location and add them to the params list
params = []
for loc in locations:
    params.append({
        'size': '600x300',
        'location': loc['location'],
        'heading': '151.78',
        'pitch': '-90',
        'key': 'YOUR_KEY'
    })

# Create a list of results objects for each request
results = google_streetview.api.results(params)



# Download images to directory 'downloads'
results.download_links(result_path)

'''import os
json_files = os.listdir(result_path)
counter = 0
for file in json_files:
    if file[-3:] == 'jpg':
        my_dest = result_path[37:] + str(counter) + ".jpg"
        my_source = result_path + '/' + file
        my_dest = result_path + '/' + my_dest
        os.rename(my_source, my_dest)
        counter += 1
'''

'''# original
import os

# Load the JSON data from file
#json_path='/content/drive/MyDrive/Image Dataset/'
json_files=os.listdir(result_path)
counter=0
for i in json_files:
  if i[-3:]=='jpg':
    my_dest =result_path[37:] + str(counter) + ".jpg"
    my_source =result_path + '/'+i
    my_dest =result_path + '/'+my_dest
    os.rename(my_source, my_dest)
    counter+= 1
'''

import json

# Open the JSON file and parse the data
with open('/content/drive/MyDrive/Sample Data/6817 Calumet Ave/metadata.json', 'r') as f:
    data = json.load(f)

# Add a new column to the JSON data
count=0
for item in data:
  item['img_name'] = result_path[35:]+str(count)+".jpg"
  count+=1
# Write the updated JSON data back to the file
with open('file.json', 'w') as f:
    json.dump(data, f)

import pandas as pd
import requests
import os
import hashlib

# Load the JSON data from file
json_path='/content/drive/MyDrive/Image Dataset/'
files=os.listdir(json_path)
counter = 1
for i in files:
  for j in os.listdir(json_path+i):
    if j[-4:]=='json':
      # Extract the name of the JSON file being read
      #file_name = j[:-5]
      df = pd.read_json(json_path+i+'/'+j)
      if 'location' in df.columns:
        # Drop the rows where location data is missing
        df = df.dropna(subset=['location'])

        # Drop the columns that are not needed
        df = df.drop(columns=['copyright', 'pano_id', 'status'])

        # Add a new column for the file name
        #df['file_name'] = file_name

        # Convert the date and time format to date format
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        # Dropping the date
        df = df.drop('date', axis=1)

        # Define a function to get the address components using the Google Maps Geocoding API
        def get_address_components(location):
          url = 'https://maps.googleapis.com/maps/api/geocode/json'
          params = {'latlng': '{},{}'.format(location['lat'], location['lng']), 'key': 'YOUR_KEY'}
          response = requests.get(url, params=params).json()
          address_components = response['results'][0]['address_components']
          country = ''
          state = ''
          street = ''
          for component in address_components:
            if 'country' in component['types']:
              country = component['long_name']
            elif 'administrative_area_level_1' in component['types']:
              state = component['long_name']
            elif 'route' in component['types']:
              street = component['long_name']
          return country, state, street

        # Apply the get_address_components function to each row of the dataframe
        df[['country', 'state', 'street']] = df['location'].apply(get_address_components).apply(pd.Series)
        df['street'] = df['street'].dropna()
        #df['image_name'] = df.apply(lambda row: '{}.jpg'.format(row['street'],counter), axis=1)
        df['Image_ID'] = df.apply(lambda row: hashlib.md5('{}{}'.format(row['street'], counter).encode()).hexdigest()+'.jpg', axis=1)
        # Save the updated dataframe to CSV


        if os.path.exists('CrackDetection_Sahith.csv'):
            df.to_csv('CrackDetection_Sahith.csv', mode='a', index=False, header=False)
        else:
            df.to_csv('CrackDetection_Sahith.csv', index=False)

from pathlib import Path
path = '/content/drive/MyDrive/Sample Data/6817 Calumet Ave'
imgs = os.listdir(path)

for i in range(len(imgs)):
  #current_path.rename(new_path)
  imgs[i].rename(df['image_name'][i])

"""***Model***"""

import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import random
from PIL import Image, ImageDraw
import os
from google.colab.patches import cv2_imshow
from torchvision.ops import RoIPool

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the data
dataset = torchvision.datasets.ImageFolder(root='/content/drive/MyDrive/Image Dataset/Cracked', transform=transform)

# Split the data

train_size = int(0.7*len(dataset))
test_size = len(dataset) - (train_size)
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
examples = enumerate(test_loader)

batch_idx, (example_data, example_targets) = next(examples)
print(batch_idx)
fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0],cmap='gray', interpolation='none')
  if example_targets[i] == 0:
        plt.title("Ground Truth: 2")
  else:
        plt.title("Ground Truth: 1")
  plt.xticks([])
  plt.yticks([])

"""MODEL


"""

'''class FasterRCNN_RPN(nn.Module):
  def __init__(self, num_classes, in_channels=512, mid_channels=512, num_anchors=9):
    super(FasterRCNN_RPN, self).__init__()

    # RPN layers
    self.num_anchors = num_anchors
    self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
    self.score_conv = nn.Conv2d(mid_channels, self.num_anchors * 2, kernel_size=1, stride=1, padding=0)
    self.bbox_conv = nn.Conv2d(mid_channels, self.num_anchors * 4, kernel_size=1, stride=1, padding=0)
    self.anchor_base = self._generate_anchor_base()

    # RoI pooling layer
    self.roi_pool = RoIPool(output_size=(7, 7), spatial_scale=1/16)

    # Fully connected layers
    self.fc1 = nn.Linear(7 * 7 * 512, 4096)
    self.fc2 = nn.Linear(4096, 4096)
    self.fc3 = nn.Linear(4096, num_classes)

  def forward(self, x, gt_boxes=None):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

  def _generate_anchor_base(self, base_size=16, ratios=[0.5, 1, 2], scales=[8, 16, 32]):
        py = base_size / 2.
        px = base_size / 2.

        anchor_base = []
        for r in ratios:
            for s in scales:
                h = s * r
                w = s * (1./r)

                x1 = px - w / 2.
                y1 = py - h / 2.
                x2 = px + w / 2.
                y2 = py + h / 2.

                anchor_base.append([x1, y1, x2, y2])

        return torch.tensor(anchor_base, dtype=torch.float32)'''

'''class FasterRCNN(nn.Module):
  def __init__(self, num_classes):
    super(FasterRCNN, self).__init__()
    self.roi_pool = RoIPool(output_size=(7, 7), spatial_scale=1/16)
    self.fc1 = nn.Linear(7 * 7 * 512, 4096)
    self.fc2 = nn.Linear(4096, 4096)
    self.fc3 = nn.Linear(4096, num_classes)

  def forward(self, x, rois):
    # x is the feature map of shape [batch_size, channels, height, width]
    # rois is a tensor of shape [num_rois, 5], where the first column is the batch index

    # pass the rois through the roi_pool module
    pooled_features = self.roi_pool(10, rois)

            # flatten the pooled features and pass them through fully connected layers
    flattened = pooled_features.view(pooled_features.size(0), -1)
    fc1_output = F.relu(self.fc1(flattened))
    fc2_output = F.relu(self.fc2(fc1_output))
    fc3_output = self.fc3(fc2_output)

    #print(type(fc3_output))
    return fc3_output'''

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VGG16(nn.Module):
    def __init__(self, num_classes=10,in_channels=512, mid_channels=512, num_anchors=9):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))

        # RPN layers
        self.num_anchors = num_anchors
        self.RPN = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.score_conv = nn.Conv2d(mid_channels, self.num_anchors * 2, kernel_size=1, stride=1, padding=0)
        self.bbox_conv = nn.Conv2d(mid_channels, self.num_anchors * 4, kernel_size=1, stride=1, padding=0)
        #self.anchor_base = self.generate_anchor_base()
        #ROI
        self.roi_pool = RoIPool(output_size=(7, 7), spatial_scale=1/16)


        self.fc1 = nn.Sequential(nn.Dropout(0.5),
            nn.Linear(6272, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc3= nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x,rois):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)

        #ROI
        out = out.reshape(out.size(0), -1)
        #pooled_features = self.roi_pool(10,rois)

            # flatten the pooled features and pass them through fully connected layers
        #flattened = pooled_features.view(pooled_features.size(0), -1)
        fc1_output = F.relu(self.fc1(out))
        fc2_output = F.relu(self.fc2(fc1_output))
        fc3_output = self.fc3(fc2_output)

        #out = self.fc1(fc3_output)
        #out=out.torch.transpose()
        #out = self.fc2(out)
        #out = self.fc3(out)
        #out= FasterRCNN_RPN(2)
        #ut= FasterRCNN(2)
        return fc3_output

num_classes = 2
num_epochs = 10
batch_size = 16
learning_rate = 0.005

model = VGG16(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)


# Train the model
total_step = len(train_loader)

total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images,2)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images,2)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        del images, labels, outputs

    print('Accuracy of the network on the test images : {} %'.format(100 * correct / total))

'''# CODE FOR LOCALIZATION

# Define the directory containing the images
image_dir = '/content/drive/MyDrive/Image Dataset/Cracked/1'

# Define the number of localizations to generate for each image
num_localizations = 2

# Iterate through each image file in the directory
for filename in os.listdir(image_dir):
    # Load the image
    image = Image.open(os.path.join(image_dir, filename))

    # Create a new image object with transparent background
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))

    # Create a draw object
    draw = ImageDraw.Draw(overlay)

    # Generate random localizations for the current image
    localizations = []
    for i in range(num_localizations):
        # Generate random coordinates for the bounding box
        x1 = random.randint(0, image.width)
        y1 = random.randint(0, image.height)
        x2 = random.randint(x1, image.width)
        y2 = random.randint(y1, image.height)

        # Add the coordinates to the localizations list
        localizations.append((x1, y1, x2, y2))

        # Draw a rectangle on the draw object for each localization
        draw.rectangle(localizations[-1], outline='red')

    # Combine the original image with the new image object
    result = Image.alpha_composite(image.convert('RGBA'), overlay)

    # Display the result image with the localizations drawn
    result.show()'''

# Define the directory containing the images
image_dir = '/content/drive/MyDrive/Image Dataset/Cracked'

# Iterate through each subdirectory in the main directory
for foldername in os.listdir(image_dir):
    folder_path = os.path.join(image_dir, foldername)

    # Define the number of localizations to generate for each image
    num_localizations = int(foldername)

    # Iterate through each image file in the subdirectory
    for filename in os.listdir(folder_path):
        # Load the image
        image = Image.open(os.path.join(folder_path, filename))

        # Create a new image object with transparent background
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))

        # Create a draw object
        draw = ImageDraw.Draw(overlay)

        # Generate random localizations for the current image
        localizations = []
        for i in range(num_localizations):
            # Generate random coordinates for the bounding box
            x1 = random.randint(0, image.width)
            y1 = random.randint(0, image.height)
            x2 = random.randint(x1, image.width)
            y2 = random.randint(y1, image.height)

            # Add the coordinates to the localizations list
            localizations.append((x1, y1, x2, y2))

            # Draw a rectangle on the draw object for each localization
            draw.rectangle(localizations[-1], outline='red')

        # Combine the original image with the new image object
        result = Image.alpha_composite(image.convert('RGBA'), overlay)

        # Display the result image with the localizations drawn
        result.show()















