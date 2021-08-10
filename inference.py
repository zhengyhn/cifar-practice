from dataset import MergedDataset
import os
import torchvision.transforms as transforms

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

if __name__ == '__main__':
    base_path = './dataset_split'
    trainset = MergedDataset(os.path.join(base_path, 'train'), train_transform, 8, 8)
    trainset.save()
    testset = MergedDataset(os.path.join(base_path, 'test'), transform, 8, 8)
    testset.save()
