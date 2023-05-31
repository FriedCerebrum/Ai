import multiprocessing
import torch
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch import nn, optim
from torch.utils.data import DataLoader

# Определение устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Определение архитектуры нейросети
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(256, 200, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(200, 200, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(200 * 16 * 16, 50)  # Обновленный размер после двух макс-пулингов
        self.fc2 = nn.Linear(50, 37)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

# Создание объекта нейросети и перемещение его на устройство
model = ConvNet().to(device)

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Определяем преобразования на изображениях
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Случайные искажения яркости, контрастности и насыщенности
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Случайные сдвиги по горизонтали и вертикали
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Создаем Dataset, используя каталоги с изображениями
train_dataset = datasets.ImageFolder(root='Training', transform=transform)
test_dataset = datasets.ImageFolder(root='Testing', transform=transform)

class_names = train_dataset.classes
print(class_names)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    # Создаем DataLoader для тренировочного и тестового наборов данных
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=16)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=16)

    dataloaders = {
        'train': train_dataloader,
        'val': test_dataloader
    }

    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(test_dataset)
    }

    num_epochs = 25

    # Сохраняем историю потерь и точности для каждой эпохи
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Установка модели в режим обучения
            else:
                model.eval()  # Установка модели в режим оценки

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Обнуление градиентов
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)

    torch.save(model.state_dict(), 'model_weights.pth')

    # Построение графиков
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), history['train_loss'], label='Train Loss')
    plt.plot(range(1, num_epochs+1), history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss history')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), history['train_acc'], label='Train Accuracy')
    plt.plot(range(1, num_epochs+1), history['val_acc'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy history')

    plt.show()
