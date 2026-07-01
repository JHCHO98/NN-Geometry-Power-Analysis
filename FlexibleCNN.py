import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import time
import json
from tqdm import tqdm

# ================== 1. 기존 모델 정의 (그대로 사용) ==================
class FlexibleCNN(nn.Module):
    def __init__(self, mode='standard'):
        super().__init__()
        if mode == 'deep_narrow':
            # 1. Deep & Narrow: 층은 깊고(5층) 채널은 좁음 (총 107,502 파라미터)
            self.features = nn.Sequential(
                nn.Conv2d(3, 24, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(24, 24, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(24, 56, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(56, 56, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(56, 92, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.classifier = nn.Linear(92 * 4 * 4, 10)
        elif mode == 'shallow_wide':
            # 2. Shallow & Wide: 층은 얕고(2층) 채널은 매우 넓음 (총 106,930 파라미터)
            self.features = nn.Sequential(
                nn.Conv2d(3, 99, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(4, 4),
                nn.Conv2d(99, 99, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.classifier = nn.Linear(99 * 4 * 4, 10)
        elif mode == 'mid_balanced':
            # 3. Mid-Balanced: 깊이(3층)와 너비 모두 균형 잡힌 구조 (총 107,502 파라미터)
            self.features = nn.Sequential(
                nn.Conv2d(3, 38, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(38, 74, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(74, 98, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.classifier = nn.Linear(98 * 4 * 4, 10)
        elif mode == 'funnel_wide_to_narrow':
            # 4. Funnel Shape: 초반에 넓고 갈수록 좁아지는 깔대기 구조 (총 107,516 파라미터)
            self.features = nn.Sequential(
                nn.Conv2d(3, 120, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(120, 70, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(70, 36, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.classifier = nn.Linear(36 * 4 * 4, 10)
        elif mode == 'uniform':
            # 5. Uniform: 모든 층의 채널 수가 동일한 균일 구조 (총 107,002 파라미터)
            self.features = nn.Sequential(
                nn.Conv2d(3, 72, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(72, 72, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(72, 72, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.classifier = nn.Linear(72 * 4 * 4, 10)
        elif mode == 'hourglass':
            # 6. Hourglass: 넓다 -> 좁다(병목) -> 다시 넓어지는 모래시계 구조 (총 107,502 파라미터)
            self.features = nn.Sequential(
                nn.Conv2d(3, 136, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(136, 28, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(28, 168, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.classifier = nn.Linear(168 * 4 * 4, 10)
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ================== 2. 학습 설정 ==================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 128
LR = 0.001
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

modes = ['deep_narrow', 'shallow_wide', 'mid_balanced', 'funnel_wide_to_narrow', 'uniform', 'hourglass']

# CIFAR-10 데이터
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0
    pbar = tqdm(loader, leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_description(f"Loss: {loss.item():.4f}")
    return running_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

# ================== 3. 메인 학습 루프 ==================
for mode in modes:
    print(f"\n{'='*20} [{mode.upper()}] 학습 시작 {'='*20}")
    
    # 파일 경로 정의
    final_path = os.path.join(CHECKPOINT_DIR, f"{mode}_final.pth")
    best_path = os.path.join(CHECKPOINT_DIR, f"{mode}_best.pth")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{mode}_checkpoint.pth")

    # 1. 이미 완료된 모델은 스킵
    if os.path.exists(final_path):
        print(f"✅ [{mode}]는 이미 학습 완료되어 스킵합니다.")
        continue

    model = FlexibleCNN(mode=mode).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    start_epoch = 0
    best_acc = 0.0

    # 2. 중간에 멈춘 기록이 있으면 이어서 학습 (Resume)
    if os.path.exists(checkpoint_path):
        print(f"🔄 [{mode}] 체크포인트 발견! 이어서 학습합니다.")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f" -> {start_epoch} 에폭부터 재개 (현재 최고 정확도: {best_acc:.2f}%)")

    # 3. 학습 시작
    for epoch in range(start_epoch, EPOCHS):
        try:
            start_time = time.time()
            train_loss = train_one_epoch(model, trainloader, criterion, optimizer)
            val_acc = evaluate(model, testloader)
            scheduler.step()
            
            epoch_time = time.time() - start_time
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {train_loss:.4f} | Acc: {val_acc:.2f}% | Best: {best_acc:.2f}% | Time: {epoch_time:.1f}s")

            # 최고 성능 모델 저장
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_path)
                print(f"  -> 🏆 최고 성능 갱신! Best 모델 저장됨.")

            # 매 에폭마다 체크포인트 저장 (불상사 방지)
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_acc': best_acc,
            }, checkpoint_path)

        except KeyboardInterrupt:
            print("\n\n🛑 사용자 중단 감지! 현재 상태 저장 후 종료합니다.")
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_acc': best_acc,
            }, checkpoint_path)
            exit()

    # 4. 해당 모델 학습 완료 처리
    torch.save(model.state_dict(), final_path)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path) # 완료되었으니 중간 체크포인트 삭제
    print(f"🎉 [{mode.upper()}] 학습 완전 종료! 최종 정확도: {best_acc:.2f}%")
    print(f" -> 최종 모델: {final_path}")

print("\n\n✅ 모든 모델 학습이 완료되었습니다!")