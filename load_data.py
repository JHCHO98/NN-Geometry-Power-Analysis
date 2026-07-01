import torch
import os
import pickle
import hashlib
import tarfile
import urllib.request
import numpy as np
import shutil

# ================== 1. 경로 및 다운로드 정보 설정 ==================
DATA_DIR = './data_cifar10'  # .pth로 변환된 최종 데이터가 저장될 폴더
CACHE_DIR = './data_cache'   # 원본 .tar.gz가 잠시 저장될 폴더

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
CIFAR10_FILENAME = "cifar-10-python.tar.gz"

LABEL_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

os.makedirs(CACHE_DIR, exist_ok=True)

# ================== 2. 에러 방어용 다운로드 함수 ==================
def download_cifar10():
    """MD5 검증을 통해 손상된 파일은 다시 다운로드합니다."""
    file_path = os.path.join(CACHE_DIR, CIFAR10_FILENAME)
    
    # 파일이 있더라도 MD5가 다르면(깨졌으면) 다시 받음
    need_download = True
    if os.path.exists(file_path):
        print(f"🔍 기존 파일 MD5 검증 중: {file_path}")
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)
        if md5.hexdigest() == CIFAR10_MD5:
            print("✅ MD5 일치. 기존 파일 사용.")
            need_download = False
        else:
            print("⚠️ 파일이 손상되었습니다. 다시 다운로드합니다.")
            os.remove(file_path)

    if need_download:
        print(f"⬇️ 다운로드 시작: {CIFAR10_URL}")
        try:
            urllib.request.urlretrieve(CIFAR10_URL, file_path)
            print("✅ 다운로드 완료.")
        except Exception as e:
            print(f"❌ 다운로드 실패: {e}")
            raise e

    # 압축 해제
    extracted_dir = os.path.join(CACHE_DIR, 'cifar-10-batches-py')
    if not os.path.exists(extracted_dir):
        print("📦 압축 해제 중...")
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(path=CACHE_DIR)
        print("✅ 압축 해제 완료.")
    return extracted_dir

# ================== 3. .pkl 변환 (저장) ==================
def convert_to_pkl():
    """CIFAR10 raw 파일들을 Numpy 배열로 변환하여 .pkl로 저장"""
    train_pkl = os.path.join(DATA_DIR, 'train.pkl')
    test_pkl = os.path.join(DATA_DIR, 'test.pkl')

    # 이미 변환되어 있으면 스킵
    if os.path.exists(train_pkl) and os.path.exists(test_pkl):
        print("✅ 이미 변환된 .pkl 파일이 존재합니다. 변환 과정을 건너뜁니다.")
        return

    print("🔄 .pkl 변환 시작...")
    extracted_dir = download_cifar10()

    def load_batch(file_path):
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            data = entry['data']
            labels = entry['labels'] if 'labels' in entry else entry['fine_labels']
        # (N, 3072) -> (N, 3, 32, 32) -> (N, 3, 32, 32) uint8
        data = data.reshape(-1, 3, 32, 32)
        return data, np.array(labels, dtype=np.int64)

    # Train (5개 배치 합치기)
    train_data, train_labels = [], []
    for i in range(1, 6):
        path = os.path.join(extracted_dir, f'data_batch_{i}')
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ 원본 파일이 없습니다: {path}")
        d, l = load_batch(path)
        train_data.append(d)
        train_labels.append(l)
    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)

    # Test
    test_path = os.path.join(extracted_dir, 'test_batch')
    test_data, test_labels = load_batch(test_path)

    # 저장 (에러 방어를 위한 try-except)
    try:
        with open(train_pkl, 'wb') as f:
            pickle.dump({'data': train_data, 'labels': train_labels}, f)
        with open(test_pkl, 'wb') as f:
            pickle.dump({'data': test_data, 'labels': test_labels}, f)
        print(f"💾 저장 완료: {train_pkl}, {test_pkl}")
    except Exception as e:
        print(f"❌ 저장 실패! '{DATA_DIR}' 폴더의 권한과 디스크 용량을 확인하세요.")
        raise e

# ================== 4. 데이터 로드 (불러오기) ==================
def load_data():
    """저장된 .pkl에서 데이터를 불러와 (data, labels) 튜플로 반환"""
    convert_to_pkl()
    
    print("📂 데이터 로드 중...")
    try:
        with open(os.path.join(DATA_DIR, 'train.pkl'), 'rb') as f:
            train_set = pickle.load(f)
        with open(os.path.join(DATA_DIR, 'test.pkl'), 'rb') as f:
            test_set = pickle.load(f)
    except Exception as e:
        print(f"❌ 로드 실패: {e}")
        raise e

    print(f"✅ 로드 완료! Train: {train_set['data'].shape}, Test: {test_set['data'].shape}")
    return train_set, test_set

# ================== 5. PyTorch Dataset 클래스 ==================
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CIFAR10Dataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.data = data_dict['data']  # (N, 3, 32, 32) uint8
        self.labels = data_dict['labels']
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # uint8 (3, 32, 32) -> PIL Image로 변환하여 transform 적용
        img = self.data[idx].transpose(1, 2, 0)  # (3, 32, 32) -> (32, 32, 3)
        from PIL import Image
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
            
        label = int(self.labels[idx])
        return img, label

def get_dataloaders(batch_size=128):
    train_set, test_set = load_data()
    
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

    train_dataset = CIFAR10Dataset(train_set, transform=transform_train)
    test_dataset = CIFAR10Dataset(test_set, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

# ================== 6. 실행 테스트 ==================
if __name__ == '__main__':
    # 데이터 로드 테스트
    train_loader, test_loader = get_dataloaders(batch_size=128)
    
    # 첫 배치 모양 확인
    for images, labels in train_loader:
        print(f"\n🎯 [검증] 이미지 배치 모양: {images.shape}")  # [128, 3, 32, 32] 예상
        print(f"🎯 [검증] 라벨 배치 모양: {labels.shape}")     # [128] 예상
        break