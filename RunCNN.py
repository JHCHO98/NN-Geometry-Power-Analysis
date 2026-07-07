import os
import time
import csv
import torch
import torch.nn as nn
from codecarbon import EmissionsTracker

# [중요] 보냈던 모델 클래스 구조 정의 그대로 유지
class FlexibleCNN(nn.Module):
    def __init__(self, mode='standard'):
        super().__init__()
        if mode == 'deep_narrow':
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.classifier = nn.Linear(64 * 4 * 4, 10)
        elif mode == 'shallow_wide':
            self.features = nn.Sequential(
                nn.Conv2d(3, 114, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(4, 4),
                nn.Conv2d(114, 114, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.classifier = nn.Linear(114 * 4 * 4, 10)
        else:
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

# 1. 벤치마크 환경 세팅 (적정기술 시나리오용 순수 CPU 강제)
device = torch.device("cpu")
print("[*] 저사양 CPU 환경 타겟 전력 측정을 시작합니다.")

# 2. 결과 저장용 CSV 파일 설정
log_file = "nn_geometry_power_log.csv"
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Model_Mode", "Layer_Index", "Layer_Type", "Loop_Count", "Duration_Sec", "Energy_Consumed_kWh"])

def log_result(mode, idx, layer_type, loops, duration, energy):
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([mode, idx, layer_type, loops, duration, energy])

# 3. 핵심 측정 함수
def measure_power(model_path, mode_name, num_loops=2000):
    print(f"\n=========================================")
    print(f"[+] 모드 [{mode_name.upper()}] 벤치마크 시작")
    print(f"=========================================")
    
    # 모델 선언 및 타겟 .pth 가중치 로드
    model = FlexibleCNN(mode=mode_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # CIFAR-10 입력 사양 고정 [배치사이즈 1, RGB 3채널, 32x32]
    dummy_input = torch.randn(1, 3, 32, 32).to(device)

    # ----------------------------------------------------
    # TRACK 1: 모델 '전체(FULL)' 추론 전력 측정
    # ----------------------------------------------------
    
    print(f"-> [Warm-up] 예비 추론 50회 진행 중...")
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)

    print(f"-> 전체 모델 {num_loops}회 순방향 추론 중...")
    tracker = EmissionsTracker(logging_logger='none')
    tracker.start()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_loops):
            _ = model(dummy_input)

    duration = time.time() - start_time
    tracker.stop()
    energy_consumed = tracker.final_emissions_data.energy_consumed
    
    log_result(mode_name, -1, "FULL_MODEL", num_loops, duration, energy_consumed)
    print(f">> [완료] 전체 전력: {energy_consumed:.8f} kWh ({duration:.2f}초)")

    # ----------------------------------------------------
    # TRACK 2: 내부 레이어별 개별 누적 측정 (추적식)
    # ----------------------------------------------------
    print("-> 레이어별 정밀 파편화 측정 시작...")
    
    # features 내부 레이어들과 classifier를 순서대로 합친 리스트 생성
    all_layers = list(model.features) + [nn.Flatten(start_dim=1), model.classifier]
    
    current_input = dummy_input

    for idx, layer in enumerate(all_layers):
        layer_type = layer.__class__.__name__
        print(f"   [{idx}] {layer_type} 추출 및 10,000회 반복 버스트...")

        # 정방향 아웃풋 형상 전사 (다음 레이어 인풋용 크기 확보)
        with torch.no_grad():
            next_input = layer(current_input)

        # 해당 레이어만 독립적으로 강제 루프 돌려 전력 소모 증폭 측정
        layer_tracker = EmissionsTracker(logging_logger='none')
        layer_tracker.start()
        layer_start_time = time.time()

        with torch.no_grad():
            for _ in range(10000):  # 레이어는 순식간에 지나가므로 만 번 루프 추천
                _ = layer(current_input)

        layer_duration = time.time() - layer_start_time
        layer_tracker.stop()
        layer_energy = layer_tracker.final_emissions_data.energy_consumed

        # CSV에 저장
        log_result(mode_name, idx, layer_type, 10000, layer_duration, layer_energy)
        
        # 바통 터치 (현재 출력이 다음 레이어의 입력이 됨)
        current_input = next_input

# 4. 실전 실행 제어부
if __name__ == "__main__":
    # ※ 타겟 컴퓨터로 옮겨둔 .pth 파일 경로 명을 적어주세요.
    # 예시로 가중치가 저장된 두 파일 명을 가정했습니다.
    file_lists=['models/deep_narrow_final.pth', 'models/funnel_wide_to_narrow_final.pth', 'models/hourglass_final.pth', 'models/mid_balanced_final.pth', 'models/shallow_wide_final.pth', 'models/uniform_final.pth']

    for filename in file_lists:
        try:
            if os.path.exists(filename):
                measure_power(model_path=filename, mode_name=filename[7:-10], num_loops=2000)
            else:
                print(f'No Such file: {filename}')
        except RuntimeError as e:
            print(e)
    

    print("\n[*] 모든 실험 완료! 'nn_geometry_power_log.csv' 결과 확인 바람.")