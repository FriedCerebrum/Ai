import torch
from torchvision import transforms
from torch import nn
from PIL import Image

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

# Загрузите веса, которые вы уже обучили
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()  # Установите модель в режим оценки

# Определите преобразования, которые нужно выполнить на изображениях
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Изменить размер изображения
    transforms.ToTensor(),  # Преобразовать изображение в тензор
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Нормализовать изображение
])

def predict(image_path):
    # Загрузите изображение, которое вы хотите классифицировать
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Применить преобразования и добавить измерение пакета
    image = image.to(device)

    # Используйте модель для получения предсказаний
    output = model(image)
    _, predicted = torch.max(output, 1)

    return predicted.item()