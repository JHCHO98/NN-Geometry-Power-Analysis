import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

class FlexibleCNN(nn.Module):
    def __init__(self, mode='standard'):
        super().__init__()

        # 입력: 3채널 (RGB) 32x32 크기 이미지
        # 출력: 10개 클래스 (CIFAR-10)

        if mode == 'deep_narrow':
            # 1. Deep & Narrow: 층은 깊지만 채널 수는 적은 구조
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2), # 16x16

                nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2), # 8x8

                nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2)  # 4x4
            )
            self.classifier = nn.Linear(64 * 4 * 4, 10)

        elif mode == 'shallow_wide':
            # 2. Shallow & Wide: 층은 얕지만 채널 수가 아주 많은 구조
            self.features = nn.Sequential(
                nn.Conv2d(3, 114, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(4, 4), # 8x8

                nn.Conv2d(114, 114, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2)  # 4x4
            )
            self.classifier = nn.Linear(114 * 4 * 4, 10)

        else:
            # 기본 베이스라인 모델
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.classifier = nn.Linear(64 * 4 * 4, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ---------------------------------------------------------
# 2. 데이터셋 로드 및 전처리 (CIFAR-10)
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("데이터셋 다운로드 중...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# ---------------------------------------------------------
# 3. 실험 환경 설정
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 연산 장치: {device}")

# 두 모드는 총 파라미터 수가 약 11만 개 내외로 거의 동일하게 통제되어 있다.
SELECTED_MODE = 'shallow_wide'
model = FlexibleCNN(mode=SELECTED_MODE).to(device)

# 총 파라미터 수 계산 및 출력 (통제 변수 확인용)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[{SELECTED_MODE.upper()}] 모델 총 파라미터 수: {total_params:,}개")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------------------------------------------------
# 4. 학습(Training) 및 시간 측정
# ---------------------------------------------------------
print("\n학습 시작...")
model.train()
start_time = time.time()

for epoch in range(30):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0

end_time = time.time()
print(f"학습 완료! 소요 시간: {end_time - start_time:.2f}초")

# ---------------------------------------------------------
# 5. 추론(Inference) 정확도 검증
# ---------------------------------------------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'10,000개 테스트 이미지에 대한 모델 정확도: {100 * correct / total:.2f}%')