# 라이브러리 가져오기 (sklearn, time, torch)
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

# 각 클래스 이름 라벨링 (수정할수도 있어서 별로도 지정)
bark_names = ['Deep','Moderate','Shallow']
leaf_names = ['down', 'up']
shape_names = ['circle', 'narrow', 'not', 'wide']
tree_names = ['long_tree', 'mid_tree', 'short_tree']
trunk_names = ['bend','side','straight']
trunk2_names = ['split', 'split_not']

#---------------------------------------------모델 평가---------------------------------------------#
# 혼동행렬 출력
def plot_confusion_matrix(ax, cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

def print_classification_report(y_true, y_pred, labels, target_names):
    report = classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0)
    print(report)

def evaluate_model(model, dataloaders, device):
    model.eval()
    
    all_bark_preds = []
    all_leaf_preds = []
    all_shape_preds = []
    all_tree_preds = []
    all_trunk_preds = []
    all_trunk2_preds = []
    
    all_bark_labels = []
    all_leaf_labels = []
    all_shape_labels = []
    all_tree_labels = []
    all_trunk_labels = []
    all_trunk2_labels = []

    with torch.no_grad():
        for inputs, labels_bark, labels_leaf, labels_shape, labels_tree, labels_trunk, labels_trunk2 in dataloaders['test']:
            inputs = inputs.to(device)
            labels_bark = labels_bark.to(device)
            labels_leaf = labels_leaf.to(device)
            labels_shape = labels_shape.to(device)
            labels_tree = labels_tree.to(device)
            labels_trunk = labels_trunk.to(device)
            labels_trunk2 = labels_trunk2.to(device)

            bark_out, leaf_out, shape_out, tree_out, trunk_out, trunk2_out = model(inputs)
            
            _, bark_preds = torch.max(bark_out,1)
            _, leaf_preds = torch.max(leaf_out, 1)
            _, shape_preds = torch.max(shape_out, 1)
            _, tree_preds = torch.max(tree_out, 1)
            _, trunk_preds = torch.max(trunk_out, 1)
            _, trunk2_preds = torch.max(trunk2_out, 1)
            
            all_bark_preds.extend(bark_preds.cpu().numpy())
            all_leaf_preds.extend(leaf_preds.cpu().numpy())
            all_shape_preds.extend(shape_preds.cpu().numpy())
            all_tree_preds.extend(tree_preds.cpu().numpy())
            all_trunk_preds.extend(trunk_preds.cpu().numpy())
            all_trunk2_preds.extend(trunk2_preds.cpu().numpy())

            all_bark_labels.extend(labels_bark.cpu().numpy())
            all_leaf_labels.extend(labels_leaf.cpu().numpy())
            all_shape_labels.extend(labels_shape.cpu().numpy())
            all_tree_labels.extend(labels_tree.cpu().numpy())
            all_trunk_labels.extend(labels_trunk.cpu().numpy())
            all_trunk2_labels.extend(labels_trunk2.cpu().numpy())


    def filter_invalid_labels(preds, labels):
        valid_idx = labels != -1
        return preds[valid_idx], labels[valid_idx]

    bark_preds, bark_labels = filter_invalid_labels(np.array(all_bark_preds), np.array(all_bark_labels))
    leaf_preds, leaf_labels = filter_invalid_labels(np.array(all_leaf_preds), np.array(all_leaf_labels))
    shape_preds, shape_labels = filter_invalid_labels(np.array(all_shape_preds), np.array(all_shape_labels))
    tree_preds, tree_labels = filter_invalid_labels(np.array(all_tree_preds), np.array(all_tree_labels))
    trunk_preds, trunk_labels = filter_invalid_labels(np.array(all_trunk_preds), np.array(all_trunk_labels))
    trunk2_preds, trunk2_labels = filter_invalid_labels(np.array(all_trunk2_preds), np.array(all_trunk2_labels))

    cm_bark = confusion_matrix(bark_labels, bark_preds, labels=range(len(bark_names)))
    cm_leaf = confusion_matrix(leaf_labels, leaf_preds, labels=range(len(leaf_names)))
    cm_shape = confusion_matrix(shape_labels, shape_preds, labels=range(len(shape_names)))
    cm_tree = confusion_matrix(tree_labels, tree_preds, labels=range(len(tree_names)))
    cm_trunk = confusion_matrix(trunk_labels, trunk_preds, labels=range(len(trunk_names)))
    cm_trunk2 = confusion_matrix(trunk2_labels, trunk2_preds, labels=range(len(trunk2_names)))

    # Print classification reports
    print("Bark Classification Report")
    print_classification_report(bark_labels, bark_preds, labels=range(len(bark_names)), target_names=bark_names)

    print("Leaf Classification Report")
    print_classification_report(leaf_labels, leaf_preds, labels=range(len(leaf_names)), target_names=leaf_names)
    
    print("Shape Classification Report")
    print_classification_report(shape_labels, shape_preds, labels=range(len(shape_names)), target_names=shape_names)
    
    print("Tree Classification Report")
    print_classification_report(tree_labels, tree_preds, labels=range(len(tree_names)), target_names=tree_names)
    
    print("Trunk Classification Report")
    print_classification_report(trunk_labels, trunk_preds, labels=range(len(trunk_names)), target_names=trunk_names)

    print("Trunk2 Classification Report")
    print_classification_report(trunk2_labels, trunk2_preds, labels=range(len(trunk2_names)), target_names=trunk2_names)


    fig, axes = plt.subplots(2, 3, figsize=(16, 10)) # fig 임시 지정 (수정 예정)

    plot_confusion_matrix(axes[0, 0], cm_bark, classes=bark_names, title='Bark Confusion Matrix', cmap=plt.cm.BuGn)
    plot_confusion_matrix(axes[0, 1], cm_leaf, classes=leaf_names, title='Leaf Confusion Matrix', cmap=plt.cm.Blues)
    plot_confusion_matrix(axes[0, 2], cm_shape, classes=shape_names, title='Shape Confusion Matrix', cmap=plt.cm.Reds)
    plot_confusion_matrix(axes[1, 0], cm_tree, classes=tree_names, title='Tree Confusion Matrix', cmap=plt.cm.Greens)
    plot_confusion_matrix(axes[1, 1], cm_trunk, classes=trunk_names, title='Trunk Confusion Matrix', cmap=plt.cm.Purples)
    plot_confusion_matrix(axes[1, 2], cm_trunk2, classes=trunk2_names, title='Trunk2 Confusion Matrix', cmap=plt.cm.PuRd)

    plt.tight_layout()
    plt.show()

#---------------------------------------------데이터셋 구성---------------------------------------------#

# 이미지와 라벨링 데이터셋 (다중 분류) : 검증셋, 테스트셋용
## 수피, 잎, 수형, 수관, 줄기, 줄기의 갈라짐 별도 지정
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels_bark, labels_leaf, labels_shape, labels_tree, labels_trunk, labels_trunk2, transform=None):
        self.image_paths = image_paths
        self.labels_bark = labels_bark
        self.labels_leaf = labels_leaf
        self.labels_shape = labels_shape
        self.labels_tree = labels_tree
        self.labels_trunk = labels_trunk
        self.labels_trunk2 = labels_trunk2
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label_bark = self.labels_bark[idx]
        label_leaf = self.labels_leaf[idx]
        label_shape = self.labels_shape[idx]
        label_tree = self.labels_tree[idx]
        label_trunk = self.labels_trunk[idx]
        label_trunk2 = self.labels_trunk2[idx]
        if self.transform:
            image = self.transform(image)
        return image, label_bark, label_leaf, label_shape, label_tree, label_trunk, label_trunk2

# 이미지와 라벨링 데이터셋 (다중 분류) : 학습셋 (3배수)
## 수피, 잎, 수형, 수관, 줄기, 줄기의 갈라짐 별도 지정
class CustomDataset_train(Dataset):
    def __init__(self, image_paths, labels_bark, labels_leaf, labels_shape, labels_tree, labels_trunk, labels_trunk2, transform=None):
        self.image_paths = image_paths
        self.labels_bark = labels_bark
        self.labels_leaf = labels_leaf
        self.labels_shape = labels_shape
        self.labels_tree = labels_tree
        self.labels_trunk = labels_trunk
        self.labels_trunk2 = labels_trunk2
        self.transform = transform

    def __len__(self):
        return 3 * len(self.image_paths)

    def __getitem__(self, idx):
        original_idx = idx % len(self.image_paths)

        image = Image.open(self.image_paths[original_idx]).convert('RGB')
        label_bark = self.labels_bark[original_idx]
        label_leaf = self.labels_leaf[original_idx]
        label_shape = self.labels_shape[original_idx]
        label_tree = self.labels_tree[original_idx]
        label_trunk = self.labels_trunk[original_idx]
        label_trunk2 = self.labels_trunk2[original_idx]

        if self.transform:
            image = self.transform(image)
        return image,label_bark, label_leaf, label_shape, label_tree, label_trunk, label_trunk2

# 다중태스크분류 함수 별도 지정 , Linear(), ReLU()로 분류기 추가 및 별도 조정
class MultiTaskModel(nn.Module):
    def __init__(self, num_ftrs, num_bark_classes, num_leaf_classes, num_shape_classes, num_tree_classes, num_trunk_classes, num_trunk2_classes):
        super(MultiTaskModel, self).__init__()
        self.shared = nn.Sequential(*list(model_ft.children())[:-1]) # 마지막 레이어 뺴고 가져오기 (순차적)
        self.fc_bark = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, num_bark_classes)
        )
        self.fc_leaf = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, num_leaf_classes)
        )
        self.fc_shape = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, num_shape_classes)
        )
        self.fc_tree = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, num_tree_classes)
        )
        self.fc_trunk = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, num_trunk_classes)
        )
        self.fc_trunk2 = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, num_trunk2_classes)
        )

    def forward(self, x):
        x = self.shared(x)
        x = x.view(x.size(0), -1)
        bark_out = self.fc_bark(x)
        leaf_out = self.fc_leaf(x)
        shape_out = self.fc_shape(x)
        tree_out = self.fc_tree(x)
        trunk_out = self.fc_trunk(x)
        trunk2_out = self.fc_trunk2(x)
        return bark_out, leaf_out, shape_out, tree_out, trunk_out, trunk2_out


# 데이터셋 로드 (jpg, jpeg, png) _ 네거티브 라벨링 사용
def load_data(data_dir):
    image_paths = []

    labels_bark = []
    labels_leaf = []
    labels_shape = []
    labels_tree = []
    labels_trunk = []
    labels_trunk2 = []

    for bark_idx, bark_name in enumerate(bark_names):
        bark_dir = os.path.join(data_dir, 'Bark', bark_name)
        if os.path.isdir(bark_dir):
            for img_name in os.listdir(bark_dir):
                img_path = os.path.join(bark_dir, img_name)
                if img_path.endswith(('.jpg', '.png')):
                    image_paths.append(img_path)
                    labels_bark.append(bark_idx)
                    labels_leaf.append(-1)
                    labels_shape.append(-1)
                    labels_tree.append(-1)
                    labels_trunk.append(-1)
                    labels_trunk2.append(-1)

    for leaf_idx, leaf_name in enumerate(leaf_names):
        leaf_dir = os.path.join(data_dir, 'Leaf', leaf_name)
        if os.path.isdir(leaf_dir):
            for img_name in os.listdir(leaf_dir):
                img_path = os.path.join(leaf_dir, img_name)
                if img_path.endswith(('.jpg', '.png')):
                    image_paths.append(img_path)
                    labels_bark.append(-1)
                    labels_leaf.append(leaf_idx)
                    labels_shape.append(-1)
                    labels_tree.append(-1)
                    labels_trunk.append(-1)
                    labels_trunk2.append(-1)

    for shape_idx, shape_name in enumerate(shape_names):
        shape_dir = os.path.join(data_dir, 'Shape', shape_name)
        if os.path.isdir(shape_dir):
            for img_name in os.listdir(shape_dir):
                img_path = os.path.join(shape_dir, img_name)
                if img_path.endswith(('.jpg', '.png')):
                    image_paths.append(img_path)
                    labels_bark.append(-1)
                    labels_leaf.append(-1)
                    labels_shape.append(shape_idx)
                    labels_tree.append(-1)
                    labels_trunk.append(-1)
                    labels_trunk2.append(-1)

    for tree_idx, tree_name in enumerate(tree_names):
        tree_dir = os.path.join(data_dir, 'Tree', tree_name)
        if os.path.isdir(tree_dir):
            for img_name in os.listdir(tree_dir):
                img_path = os.path.join(tree_dir, img_name)
                if img_path.endswith(('.jpg', '.png')):
                    image_paths.append(img_path)
                    labels_bark.append(-1)
                    labels_leaf.append(-1)
                    labels_shape.append(-1)
                    labels_tree.append(tree_idx)
                    labels_trunk.append(-1)
                    labels_trunk2.append(-1)

    for trunk_idx, trunk_name in enumerate(trunk_names):
        trunk_dir = os.path.join(data_dir, 'Trunk', trunk_name)
        if os.path.isdir(trunk_dir):
            for img_name in os.listdir(trunk_dir):
                img_path = os.path.join(trunk_dir, img_name)
                if img_path.endswith(('.jpg', '.png')):
                    image_paths.append(img_path)
                    labels_bark.append(-1)
                    labels_leaf.append(-1)
                    labels_shape.append(-1)
                    labels_tree.append(-1)
                    labels_trunk.append(trunk_idx)
                    labels_trunk2.append(-1)

    for trunk2_idx, trunk2_name in enumerate(trunk2_names):
        trunk2_dir = os.path.join(data_dir, 'Trunk2', trunk2_name)
        if os.path.isdir(trunk2_dir):
            for img_name in os.listdir(trunk2_dir):
                img_path = os.path.join(trunk2_dir, img_name)
                if img_path.endswith(('.jpg', '.png')):
                    image_paths.append(img_path)
                    labels_bark.append(-1)
                    labels_leaf.append(-1)
                    labels_shape.append(-1)
                    labels_tree.append(-1)
                    labels_trunk.append(-1)
                    labels_trunk2.append(trunk2_idx)                

    return image_paths, labels_bark, labels_leaf, labels_shape, labels_tree, labels_trunk, labels_trunk2

# 데이터 증강 _ 파이토치 튜토리얼 변형
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

# 모델 학습 함수_다중 테스크 분류 (디폴트 : 에포크25, 배치사이즈 32)
def train_multi_task_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=25):
    since = time.time()

    loss_weights = {
    'bark': 1.0,
    'leaf': 1.0,
    'shape': 1.0,
    'tree': 1.0,
    'trunk': 1.0,
    'trunk2': 1.0
    }

    best_acc_bark = 0.0
    best_acc_leaf = 0.0
    best_acc_shape = 0.0
    best_acc_tree = 0.0
    best_acc_trunk = 0.0
    best_acc_trunk2 = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_bark_corrects = 0
            running_leaf_corrects = 0
            running_shape_corrects = 0
            running_tree_corrects = 0
            running_trunk_corrects = 0
            running_trunk2_corrects = 0

            for inputs, labels_bark, labels_leaf, labels_shape, labels_tree, labels_trunk, labels_trunk2 in dataloaders[phase]:
                inputs = inputs.to(device)
                labels_bark = labels_bark.to(device)
                labels_leaf = labels_leaf.to(device)
                labels_shape = labels_shape.to(device)
                labels_tree = labels_tree.to(device)
                labels_trunk = labels_trunk.to(device)
                labels_trunk2 = labels_trunk2.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    bark_out, leaf_out, shape_out, tree_out, trunk_out, trunk2_out = model(inputs)
                    _, bark_preds = torch.max(bark_out, 1)
                    _, leaf_preds = torch.max(leaf_out, 1)
                    _, shape_preds = torch.max(shape_out, 1)
                    _, tree_preds = torch.max(tree_out, 1)
                    _, trunk_preds = torch.max(trunk_out, 1)
                    _, trunk2_preds = torch.max(trunk2_out, 1)

                    # 가중치 별도 계산 
                    loss = 0
                    if labels_bark.ne(-1).sum() > 0:
                        loss_bark = criterion(bark_out[labels_bark != -1], labels_bark[labels_bark != -1])
                        loss += loss_weights['bark'] * loss_bark
                    if labels_leaf.ne(-1).sum() > 0:
                        loss_leaf = criterion(leaf_out[labels_leaf != -1], labels_leaf[labels_leaf != -1])
                        loss += loss_weights['leaf'] * loss_leaf
                    if labels_shape.ne(-1).sum() > 0:
                        loss_shape = criterion(shape_out[labels_shape != -1], labels_shape[labels_shape != -1])
                        loss += loss_weights['shape'] * loss_shape
                    if labels_tree.ne(-1).sum() > 0:
                        loss_tree = criterion(tree_out[labels_tree != -1], labels_tree[labels_tree != -1])
                        loss += loss_weights['tree'] * loss_tree
                    if labels_trunk.ne(-1).sum() > 0:
                        loss_trunk = criterion(trunk_out[labels_trunk != -1], labels_trunk[labels_trunk != -1])
                        loss += loss_weights['trunk'] * loss_trunk
                    if labels_trunk2.ne(-1).sum() > 0:
                        loss_trunk2 = criterion(trunk2_out[labels_trunk2 != -1], labels_trunk2[labels_trunk2 != -1])
                        loss += loss_weights['trunk2'] * loss_trunk2     

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_bark_corrects += torch.sum((bark_preds == labels_bark.data) & (labels_bark != -1))               
                running_leaf_corrects += torch.sum((leaf_preds == labels_leaf.data) & (labels_leaf != -1))
                running_shape_corrects += torch.sum((shape_preds == labels_shape.data) & (labels_shape != -1))
                running_tree_corrects += torch.sum((tree_preds == labels_tree.data) & (labels_tree != -1))
                running_trunk_corrects += torch.sum((trunk_preds == labels_trunk.data) & (labels_trunk != -1))
                running_trunk2_corrects += torch.sum((trunk2_preds == labels_trunk2.data) & (labels_trunk2 != -1))

            epoch_loss = running_loss / dataset_sizes[phase]

            epoch_bark_acc = running_bark_corrects.double() / (dataset_sizes[phase] - (labels_bark == -1).sum())            
            epoch_leaf_acc = running_leaf_corrects.double() / (dataset_sizes[phase] - (labels_leaf == -1).sum())
            epoch_shape_acc = running_shape_corrects.double() / (dataset_sizes[phase] - (labels_shape == -1).sum())
            epoch_tree_acc = running_tree_corrects.double() / (dataset_sizes[phase] - (labels_tree == -1).sum())
            epoch_trunk_acc = running_trunk_corrects.double() / (dataset_sizes[phase] - (labels_trunk == -1).sum())
            epoch_trunk2_acc = running_trunk2_corrects.double() / (dataset_sizes[phase] - (labels_trunk2 == -1).sum())

            # 정확도 계산이랑 다르게 나옴 x4로 이해할것
            print(f'{phase} Loss: {epoch_loss:.4f} | Bark Acc: {epoch_bark_acc:.4f} | Leaf Acc: {epoch_leaf_acc:.4f} | Shape Acc: {epoch_shape_acc:.4f} | Tree Acc: {epoch_tree_acc:.4f} | Trunk Acc: {epoch_trunk_acc:.4f} | Trunk2 Acc: {epoch_trunk2_acc:.4f}')

            if phase == 'val' and (epoch_bark_acc > best_acc_bark or epoch_leaf_acc > best_acc_leaf or epoch_shape_acc > best_acc_shape or epoch_tree_acc > best_acc_tree or epoch_trunk_acc > best_acc_trunk or epoch_trunk2_acc > best_acc_trunk2):
                best_acc_bark = epoch_bark_acc
                best_acc_leaf = epoch_leaf_acc
                best_acc_shape = epoch_shape_acc
                best_acc_tree = epoch_tree_acc
                best_acc_trunk = epoch_trunk_acc
                best_acc_trunk2 = epoch_trunk2_acc
                best_model_params_path = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since

    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Bark Acc: {best_acc_bark:.4f}')
    print(f'Best val Leaf Acc: {best_acc_leaf:.4f}')
    print(f'Best val Shape Acc: {best_acc_shape:.4f}')
    print(f'Best val Tree Acc: {best_acc_tree:.4f}')
    print(f'Best val Trunk Acc: {best_acc_trunk:.4f}')
    print(f'Best val Trunk2 Acc: {best_acc_trunk2:.4f}\n')

    model.load_state_dict(best_model_params_path)
    return model

#---------------------------------------------메인 함수---------------------------------------------#
def main():
    data_dir = 'dataset'
    image_paths, labels_bark, labels_leaf, labels_shape, labels_tree, labels_trunk, labels_trunk2 = load_data(data_dir)
    
    # 데이터셋 분할 학습 0.8  검증 0.1 테스트 0.1
    train_paths, temp_paths, train_labels_bark, temp_labels_bark, train_labels_leaf, temp_labels_leaf, train_labels_shape, temp_labels_shape, train_labels_tree, temp_labels_tree, train_labels_trunk, temp_labels_trunk, train_labels_trunk2, temp_labels_trunk2 = train_test_split(
        image_paths,labels_bark, labels_leaf, labels_shape, labels_tree, labels_trunk, labels_trunk2, test_size=0.2, random_state=100)
    
    val_paths, test_paths,val_labels_bark, test_labels_bark, val_labels_leaf, test_labels_leaf, val_labels_shape, test_labels_shape, val_labels_tree, test_labels_tree, val_labels_trunk, test_labels_trunk, val_labels_trunk2, test_labels_trunk2 = train_test_split(
        temp_paths, temp_labels_bark, temp_labels_leaf, temp_labels_shape, temp_labels_tree, temp_labels_trunk, temp_labels_trunk2, test_size=0.5, random_state=100)
    
    # 데이터셋 구성
    train_dataset = CustomDataset_train(train_paths,train_labels_bark, train_labels_leaf, train_labels_shape, train_labels_tree, train_labels_trunk, train_labels_trunk2, transform=data_transforms['train'])
    val_dataset = CustomDataset(val_paths, val_labels_bark, val_labels_leaf, val_labels_shape, val_labels_tree, val_labels_trunk,val_labels_trunk2, transform=data_transforms['val'])
    test_dataset = CustomDataset(test_paths, test_labels_bark, test_labels_leaf, test_labels_shape, test_labels_tree, test_labels_trunk, test_labels_trunk2, transform=data_transforms['test'])

    # 데이터 로더(학습셋만 셔플, num_worker=4 디폴트, 나머지 적용 x)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}
    
    ## CUDA 사용
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 캡슐 참조
    global model_ft
    model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1) 

    # 학습 정보 구성
    num_ftrs = model_ft.fc.in_features

    num_bark_classes = len(bark_names)    
    num_leaf_classes = len(leaf_names)
    num_shape_classes = len(shape_names)
    num_tree_classes = len(tree_names)
    num_trunk_classes = len(trunk_names)
    num_trunk2_classes = len(trunk_names)

    model_ft = MultiTaskModel(num_ftrs, num_bark_classes, num_leaf_classes, num_shape_classes, num_tree_classes, num_trunk_classes, num_trunk2_classes)
    model_ft = model_ft.to(device)

    # 손실 함수 및 옵티마이저 설정 : 학습률 0.001 이외 튜토리얼 기본 값
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # 모델 학습 (에포크 100 구성)
    model_ft = train_multi_task_model(model_ft, criterion, optimizer_ft, dataloaders, dataset_sizes, device, num_epochs=101)

    torch.save(model_ft.state_dict(), 'resnet50_multitask_model_ver4.pth')

    evaluate_model(model_ft, dataloaders, device)

if __name__ == '__main__':
    main()
