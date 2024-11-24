import torch
from torch import nn  
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from timeit import default_timer as timer 
import cv2 as cv2
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

#Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

#Setting up Path
data_path = Path("data/")
image_path = data_path / "Gesture"
if image_path.is_dir():
    print(f"{image_path} directory exist")
else:
    print(f"{image_path} directory does not exist")   

def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'")


train_dir = image_path / "train"
test_dir = image_path / "validation"


data_transform = transforms.Compose([
    transforms.Resize(size=(100, 100)),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomRotation(degrees=(0,30)),
    transforms.ToTensor(),
])
train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform,
                                  target_transform=None)
test_data = datasets.ImageFolder(root=test_dir,
                                  transform=data_transform)
class_names = train_data.classes
print(class_names)
BATCH_SIZE=8
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True
                              )
test_dataloader = DataLoader(dataset=test_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
class GestureRecognitionModel(nn.Module):
    def __init__(self):
        super(GestureRecognitionModel, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 5, 1, 2),  
            torch.nn.MaxPool2d(2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(20000, 64),
            torch.nn.Sigmoid(),
            torch.nn.Dropout(0.5), 
            torch.nn.Linear(64, 8)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        res = conv2_out.view(conv2_out.size(0), -1)
        out = self.dense(res)
        return out
# Setup loss and optimizer
model = GestureRecognitionModel().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), 
                             lr=1e-3)
trainloss = []
testloss = []
Trainaccuracy = []
testaccuracy = []
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

# Measure time
train_time_start_model_1 = timer()
batches_per_epoch = len(train_dataloader) 
# Train and test model 
epochs = 12
model.train()
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")
    ### Training
    train_loss,train_acc = 0,0
    # Add a loop to loop through training batches
    for batch, (X,y) in enumerate(train_dataloader):
        X = X.to(device)
        y = y.to(device)
        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss # accumulatively add up the loss per epoch 
        train_acc += accuracy_fn(y_true=y,
                                y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels
        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Divide total train loss by length of train dataloader (average loss per batch per epoch)
    train_loss /= len(train_dataloader)
    trainloss.append(torch.tensor(train_loss).detach().cpu().numpy())
    train_acc /= len(train_dataloader)
    Trainaccuracy.append(torch.tensor(train_acc).detach().cpu().numpy())
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
    
    ### Testing
    # Setup variables for accumulatively adding up loss and accuracy 
    test_loss, test_acc = 0, 0 
    model.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            X = X.to(device) 
            y = y.to(device)
            # 1. Forward pass
            test_pred = model(X)
            # 2. Calculate loss (accumatively)
            test_loss += loss_fn(test_pred, y) # accumulatively add up the loss per epoch

            # 3. Calculate accuracy (preds need to be same as y_true)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
        
        # Calculations on test metrics need to happen inside torch.inference_mode()
        # Divide total test loss by length of test dataloader (per batch)
        test_loss /= len(test_dataloader)
        testloss.append(torch.tensor(test_loss).detach().cpu().numpy())
        # Divide total accuracy by length of test dataloader (per batch)
        test_acc /= len(test_dataloader)
        testaccuracy.append(torch.tensor(test_acc).detach().cpu().numpy())
    ## Print out what's happening
    print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")



epochss = range(len(trainloss))
# Setup a plot 
plt.figure(figsize=(15, 7))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(epochss, trainloss, label='train_loss')
plt.plot(epochss, testloss, label='test_loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(epochss, Trainaccuracy, label='train_accuracy')
plt.plot(epochss, testaccuracy, label='test_accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()
# Calculate training time      
train_time_end_model_2 = timer()
total_train_time_model_2 = print_train_time(start=train_time_start_model_1,
                                           end=train_time_end_model_2,
                                           device=device)


torch.save(model.state_dict(), 'model.pth')


C_pred = []
C_true = []

# iterate over test data
for inputs, labels in test_dataloader:
        inputs = inputs.to(device) 
        labels = labels.to(device)
        output = model(inputs) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        C_pred.extend(output) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        C_true.extend(labels) # Save Truth

# Build confusion matrix
cf_matrix = confusion_matrix(C_true, C_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in class_names],
                     columns = [i for i in class_names])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('output3.png')
