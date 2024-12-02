import os
import time
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
import copy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

leaf_names = ['No', 'Yes']
shape_names = ['conical_circle', 'conical_narrow', 'conical_not', 'conical_wide']
tree_names = ['long_tree', 'mid_tree', 'short_tree']
trunk_names = ['Not_split', 'split']

def plot_confusion_matrix(ax, cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

def print_classification_report(y_true, y_pred, labels, target_names):
    report = classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0)
    print(report)

def evaluate_model(model, dataloaders, device):
    model.eval()
    
    all_leaf_preds = []
    all_shape_preds = []
    all_tree_preds = []
    all_trunk_preds = []
    
    all_leaf_labels = []
    all_shape_labels = []
    all_tree_labels = []
    all_trunk_labels = []

    with torch.no_grad():
        for inputs, labels_leaf, labels_shape, labels_tree, labels_trunk in dataloaders['test']:
            inputs = inputs.to(device)
            labels_leaf = labels_leaf.to(device)
            labels_shape = labels_shape.to(device)
            labels_tree = labels_tree.to(device)
            labels_trunk = labels_trunk.to(device)

            leaf_out, shape_out, tree_out, trunk_out = model(inputs)
            
            _, leaf_preds = torch.max(leaf_out, 1)
            _, shape_preds = torch.max(shape_out, 1)
            _, tree_preds = torch.max(tree_out, 1)
            _, trunk_preds = torch.max(trunk_out, 1)
            
            all_leaf_preds.extend(leaf_preds.cpu().numpy())
            all_shape_preds.extend(shape_preds.cpu().numpy())
            all_tree_preds.extend(tree_preds.cpu().numpy())
            all_trunk_preds.extend(trunk_preds.cpu().numpy())
            
            all_leaf_labels.extend(labels_leaf.cpu().numpy())
            all_shape_labels.extend(labels_shape.cpu().numpy())
            all_tree_labels.extend(labels_tree.cpu().numpy())
            all_trunk_labels.extend(labels_trunk.cpu().numpy())

    def filter_invalid_labels(preds, labels):
        valid_idx = labels != -1
        return preds[valid_idx], labels[valid_idx]

    leaf_preds, leaf_labels = filter_invalid_labels(np.array(all_leaf_preds), np.array(all_leaf_labels))
    shape_preds, shape_labels = filter_invalid_labels(np.array(all_shape_preds), np.array(all_shape_labels))
    tree_preds, tree_labels = filter_invalid_labels(np.array(all_tree_preds), np.array(all_tree_labels))
    trunk_preds, trunk_labels = filter_invalid_labels(np.array(all_trunk_preds), np.array(all_trunk_labels))

    cm_leaf = confusion_matrix(leaf_labels, leaf_preds, labels=range(len(leaf_names)))
    cm_shape = confusion_matrix(shape_labels, shape_preds, labels=range(len(shape_names)))
    cm_tree = confusion_matrix(tree_labels, tree_preds, labels=range(len(tree_names)))
    cm_trunk = confusion_matrix(trunk_labels, trunk_preds, labels=range(len(trunk_names)))
  
    print("Leaf Classification Report")
    print_classification_report(leaf_labels, leaf_preds, labels=range(len(leaf_names)), target_names=leaf_names)
    
    print("Shape Classification Report")
    print_classification_report(shape_labels, shape_preds, labels=range(len(shape_names)), target_names=shape_names)
    
    print("Tree Classification Report")
    print_classification_report(tree_labels, tree_preds, labels=range(len(tree_names)), target_names=tree_names)
    
    print("Trunk Classification Report")
    print_classification_report(trunk_labels, trunk_preds, labels=range(len(trunk_names)), target_names=trunk_names)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    plot_confusion_matrix(axes[0, 0], cm_leaf, classes=leaf_names, title='Leaf Confusion Matrix', cmap=plt.cm.Blues)
    plot_confusion_matrix(axes[0, 1], cm_shape, classes=shape_names, title='Shape Confusion Matrix', cmap=plt.cm.Reds)
    plot_confusion_matrix(axes[1, 0], cm_tree, classes=tree_names, title='Tree Confusion Matrix', cmap=plt.cm.Greens)
    plot_confusion_matrix(axes[1, 1], cm_trunk, classes=trunk_names, title='Trunk Confusion Matrix', cmap=plt.cm.Purples)

    plt.show()

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels_leaf, labels_shape, labels_tree, labels_trunk, transform=None):
        self.image_paths = image_paths
        self.labels_leaf = labels_leaf
        self.labels_shape = labels_shape
        self.labels_tree = labels_tree
        self.labels_trunk = labels_trunk
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label_leaf = self.labels_leaf[idx]
        label_shape = self.labels_shape[idx]
        label_tree = self.labels_tree[idx]
        label_trunk = self.labels_trunk[idx]
        if self.transform:
            image = self.transform(image)
        return image, label_leaf, label_shape, label_tree, label_trunk

class MultiTaskModel(nn.Module):
    def __init__(self, num_ftrs, num_leaf_classes, num_shape_classes, num_tree_classes, num_trunk_classes):
        super(MultiTaskModel, self).__init__()
        self.shared = nn.Sequential(*list(model_ft.children())[:-1])
        self.fc_leaf = nn.Linear(num_ftrs, num_leaf_classes)
        self.fc_shape = nn.Linear(num_ftrs, num_shape_classes)
        self.fc_tree = nn.Linear(num_ftrs, num_tree_classes)
        self.fc_trunk = nn.Linear(num_ftrs, num_trunk_classes)

    def forward(self, x):
        x = self.shared(x)
        x = x.view(x.size(0), -1)
        leaf_out = self.fc_leaf(x)
        shape_out = self.fc_shape(x)
        tree_out = self.fc_tree(x)
        trunk_out = self.fc_trunk(x)
        return leaf_out, shape_out, tree_out, trunk_out

def load_data(data_dir):
    image_paths = []
    labels_leaf = []
    labels_shape = []
    labels_tree = []
    labels_trunk = []

    for leaf_idx, leaf_name in enumerate(leaf_names):
        leaf_dir = os.path.join(data_dir, 'leaf', leaf_name)
        if os.path.isdir(leaf_dir):
            for img_name in os.listdir(leaf_dir):
                img_path = os.path.join(leaf_dir, img_name)
                if img_path.endswith(('.jpg', '.png')):
                    image_paths.append(img_path)
                    labels_leaf.append(leaf_idx)
                    labels_shape.append(-1)
                    labels_tree.append(-1)
                    labels_trunk.append(-1)

    for shape_idx, shape_name in enumerate(shape_names):
        shape_dir = os.path.join(data_dir, 'shape', shape_name)
        if os.path.isdir(shape_dir):
            for img_name in os.listdir(shape_dir):
                img_path = os.path.join(shape_dir, img_name)
                if img_path.endswith(('.jpg', '.png')):
                    image_paths.append(img_path)
                    labels_leaf.append(-1)
                    labels_shape.append(shape_idx)
                    labels_tree.append(-1)
                    labels_trunk.append(-1)

    for tree_idx, tree_name in enumerate(tree_names):
        tree_dir = os.path.join(data_dir, 'tree', tree_name)
        if os.path.isdir(tree_dir):
            for img_name in os.listdir(tree_dir):
                img_path = os.path.join(tree_dir, img_name)
                if img_path.endswith(('.jpg', '.png')):
                    image_paths.append(img_path)
                    labels_leaf.append(-1)
                    labels_shape.append(-1)
                    labels_tree.append(tree_idx)
                    labels_trunk.append(-1)

    for trunk_idx, trunk_name in enumerate(trunk_names):
        trunk_dir = os.path.join(data_dir, 'trunk', trunk_name)
        if os.path.isdir(trunk_dir):
            for img_name in os.listdir(trunk_dir):
                img_path = os.path.join(trunk_dir, img_name)
                if img_path.endswith(('.jpg', '.png')):
                    image_paths.append(img_path)
                    labels_leaf.append(-1)
                    labels_shape.append(-1)
                    labels_tree.append(-1)
                    labels_trunk.append(trunk_idx)

    return image_paths, labels_leaf, labels_shape, labels_tree, labels_trunk

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def train_multi_task_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=25):
    since = time.time()

    best_acc_leaf = 0.0
    best_acc_shape = 0.0
    best_acc_tree = 0.0
    best_acc_trunk = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_leaf_corrects = 0
            running_shape_corrects = 0
            running_tree_corrects = 0
            running_trunk_corrects = 0

            for inputs, labels_leaf, labels_shape, labels_tree, labels_trunk in dataloaders[phase]:
                inputs = inputs.to(device)
                labels_leaf = labels_leaf.to(device)
                labels_shape = labels_shape.to(device)
                labels_tree = labels_tree.to(device)
                labels_trunk = labels_trunk.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    leaf_out, shape_out, tree_out, trunk_out = model(inputs)
                    _, leaf_preds = torch.max(leaf_out, 1)
                    _, shape_preds = torch.max(shape_out, 1)
                    _, tree_preds = torch.max(tree_out, 1)
                    _, trunk_preds = torch.max(trunk_out, 1)

                    loss = 0
                    if labels_leaf.ne(-1).sum() > 0:
                        loss_leaf = criterion(leaf_out[labels_leaf != -1], labels_leaf[labels_leaf != -1])
                        loss += loss_leaf
                    if labels_shape.ne(-1).sum() > 0:
                        loss_shape = criterion(shape_out[labels_shape != -1], labels_shape[labels_shape != -1])
                        loss += loss_shape
                    if labels_tree.ne(-1).sum() > 0:
                        loss_tree = criterion(tree_out[labels_tree != -1], labels_tree[labels_tree != -1])
                        loss += loss_tree
                    if labels_trunk.ne(-1).sum() > 0:
                        loss_trunk = criterion(trunk_out[labels_trunk != -1], labels_trunk[labels_trunk != -1])
                        loss += loss_trunk

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_leaf_corrects += torch.sum((leaf_preds == labels_leaf.data) & (labels_leaf != -1))
                running_shape_corrects += torch.sum((shape_preds == labels_shape.data) & (labels_shape != -1))
                running_tree_corrects += torch.sum((tree_preds == labels_tree.data) & (labels_tree != -1))
                running_trunk_corrects += torch.sum((trunk_preds == labels_trunk.data) & (labels_trunk != -1))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_leaf_acc = running_leaf_corrects.double() / (dataset_sizes[phase] - (labels_leaf == -1).sum())
            epoch_shape_acc = running_shape_corrects.double() / (dataset_sizes[phase] - (labels_shape == -1).sum())
            epoch_tree_acc = running_tree_corrects.double() / (dataset_sizes[phase] - (labels_tree == -1).sum())
            epoch_trunk_acc = running_trunk_corrects.double() / (dataset_sizes[phase] - (labels_trunk == -1).sum())

            print(f'{phase} Loss: {epoch_loss:.4f} | Leaf Acc: {epoch_leaf_acc:.4f} | Shape Acc: {epoch_shape_acc:.4f} | Tree Acc: {epoch_tree_acc:.4f} | Trunk Acc: {epoch_trunk_acc:.4f}')

            if phase == 'val' and (epoch_leaf_acc > best_acc_leaf or epoch_shape_acc > best_acc_shape or epoch_tree_acc > best_acc_tree or epoch_trunk_acc > best_acc_trunk):
                best_acc_leaf = epoch_leaf_acc
                best_acc_shape = epoch_shape_acc
                best_acc_tree = epoch_tree_acc
                best_acc_trunk = epoch_trunk_acc
                best_model_params_path = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Leaf Acc: {best_acc_leaf:.4f}')
    print(f'Best val Shape Acc: {best_acc_shape:.4f}')
    print(f'Best val Tree Acc: {best_acc_tree:.4f}')
    print(f'Best val Trunk Acc: {best_acc_trunk:.4f}\n')

    model.load_state_dict(best_model_params_path)
    return model

def main():
    data_dir = 'dataset'
    image_paths, labels_leaf, labels_shape, labels_tree, labels_trunk = load_data(data_dir)

    train_paths, temp_paths, train_labels_leaf, temp_labels_leaf, train_labels_shape, temp_labels_shape, train_labels_tree, temp_labels_tree, train_labels_trunk, temp_labels_trunk = train_test_split(
        image_paths, labels_leaf, labels_shape, labels_tree, labels_trunk, test_size=0.2, random_state=100)
    
    val_paths, test_paths, val_labels_leaf, test_labels_leaf, val_labels_shape, test_labels_shape, val_labels_tree, test_labels_tree, val_labels_trunk, test_labels_trunk = train_test_split(
        temp_paths, temp_labels_leaf, temp_labels_shape, temp_labels_tree, temp_labels_trunk, test_size=0.5, random_state=100)


    train_dataset = CustomDataset(train_paths, train_labels_leaf, train_labels_shape, train_labels_tree, train_labels_trunk, transform=data_transforms['train'])
    val_dataset = CustomDataset(val_paths, val_labels_leaf, val_labels_shape, val_labels_tree, val_labels_trunk, transform=data_transforms['val'])
    test_dataset = CustomDataset(test_paths, test_labels_leaf, test_labels_shape, test_labels_tree, test_labels_trunk, transform=data_transforms['test'])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,  num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,  num_workers=4)

    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    global model_ft
    model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1) 

    num_ftrs = model_ft.fc.in_features
    num_leaf_classes = len(leaf_names)
    num_shape_classes = len(shape_names)
    num_tree_classes = len(tree_names)
    num_trunk_classes = len(trunk_names)

    model_ft = MultiTaskModel(num_ftrs, num_leaf_classes, num_shape_classes, num_tree_classes, num_trunk_classes)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    model_ft = train_multi_task_model(model_ft, criterion, optimizer_ft, dataloaders, dataset_sizes, device, num_epochs=2)

    torch.save(model_ft.state_dict(), 'resnet50_multitask_model.pth')

    evaluate_model(model_ft, dataloaders, device)


if __name__ == '__main__':
    main()
