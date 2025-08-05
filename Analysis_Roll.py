import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import matplotlib.pyplot as plt

# --- 분석 파라미터 설정 (이곳에서 모든 값을 조정하세요) ---
# 분석할 폴더의 전체 경로.
# 예: r"C:\Users\user\Desktop\qtm_export_20250801_112825"
TARGET_DIRECTORY = r"C:\Users\user\Desktop\real-Time\qtm_export_20250801_112825"

# 데이터 샘플링 속도 (Hz).
SAMPLING_RATE = 120.0

# 데이터 분석 시 평활화를 위한 윈도우 크기. 클수록 그래프가 부드러워짐.
SMOOTHING_WINDOW = 30

# 움직임 감지 시 최소 상승 값 (degree).
RISE_THRESHOLD = 2.5

# 안정 구간 탐색 시 허용 오차 (degree).
STABILITY_TOLERANCE = 1.0

# 안정 상태로 판단하기 위한 최소 프레임 수.
MIN_STABLE_FRAMES = int(SAMPLING_RATE * 0.5) # 0.5초

# 동적 기준선 탐색 시 '0도 근처'로 판단할 허용 오차 (degree).
BASELINE_ZERO_TOLERANCE = 2.0

# 유효한 움직임으로 판단할 'Peak부터 End까지'의 최소 소요 시간 (초).
MIN_PEAK_TO_END_DURATION_SECONDS = 20.0
# --------------------------------------------------------

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_debug_plot(df: pd.DataFrame, baseline: float, threshold: float, humps: list, output_path: Path):
    """
    분석 과정을 시각화하는 디버그 플롯을 생성하고 저장합니다.
    """
    plt.figure(figsize=(20, 8))
    plt.plot(df['Time'], df['Roll'], color='lightgray', label='Original Roll')
    plt.plot(df['Time'], df['Roll_Smooth'], color='blue', label='Smoothed Roll (Analysis Target)')
    plt.axhline(y=baseline, color='orange', linestyle='--', label=f'Dynamic Baseline ({baseline:.2f})')
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Rise Threshold ({threshold:.2f})')
    
    for i, hump in enumerate(humps):
        start_time = hump.get('start_frame', 0) / SAMPLING_RATE
        peak_time = hump.get('peak_frame', 0) / SAMPLING_RATE
        end_time = hump.get('end_frame', 0) / SAMPLING_RATE
        plt.axvline(x=start_time, color='green', linestyle=':', label=f'Hump {i+1} Start' if i==0 else "")
        plt.axvline(x=peak_time, color='purple', linestyle=':', label=f'Hump {i+1} Peak' if i==0 else "")
        plt.axvline(x=end_time, color='black', linestyle=':', label=f'Hump {i+1} End' if i==0 else "")

    plt.title(f'Roll Analysis Debug Plot - {output_path.stem}')
    plt.xlabel('Time (s)')
    plt.ylabel('Roll (degrees)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    plot_filename = output_path.parent / f"{output_path.stem}_debug_plot.png"
    plt.savefig(plot_filename)
    plt.close()
    logging.info(f"디버그 플롯이 '{plot_filename}'에 저장되었습니다.")


def analyze_roll_data(file_path: Path, output_dir: Path):
    """
    하나의 Rigid Body CSV 파일에서 Roll 데이터를 분석하고, 디버그 플롯을 생성합니다.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logging.error(f"파일을 찾을 수 없습니다: {file_path}"); return

    df['Roll'] = df['Roll'] * -1

    if 'Roll' not in df.columns or 'Frame' not in df.columns:
        logging.warning(f"'{file_path.name}' 파일에 'Roll' 또는 'Frame' 열이 없어 건너뜁니다."); return

    df['Roll_Smooth'] = df['Roll'].rolling(window=SMOOTHING_WINDOW, min_periods=1, center=True).mean()
    df['Time'] = df['Frame'] / SAMPLING_RATE

    baseline = None
    for i in range(len(df) - MIN_STABLE_FRAMES):
        window = df['Roll_Smooth'].iloc[i : i + MIN_STABLE_FRAMES]
        if (window.abs() < BASELINE_ZERO_TOLERANCE).all():
            baseline = window.mean()
            logging.info(f"동적 기준선 탐색 성공. {i/SAMPLING_RATE:.2f}초 부근에서 안정 구간을 찾아 기준선을 {baseline:.2f}(으)로 설정합니다.")
            break
    
    if baseline is None:
        baseline = df['Roll_Smooth'].iloc[:int(SAMPLING_RATE)].mean()
        logging.warning(f"안정적인 0도 부근을 찾지 못했습니다. 파일 시작 1초를 기준으로 강제 설정합니다 (기준선: {baseline:.2f}).")

    rise_threshold_value = baseline + RISE_THRESHOLD

    state = "IDLE"; all_humps = []; current_hump = {}

    for i in range(1, len(df)):
        prev_val, curr_val = df['Roll_Smooth'].iloc[i-1], df['Roll_Smooth'].iloc[i]
        
        if state == "IDLE":
            if curr_val > rise_threshold_value and curr_val > prev_val:
                state = "RISING"; current_hump = {'start_frame': df['Frame'].iloc[i]}
        elif state == "RISING":
            if curr_val < prev_val:
                state = "PEAK_STABILIZING"
                current_hump.update({'peak_frame': df['Frame'].iloc[i-1], 'peak_value': prev_val, 'stabilization_start_frame': -1})
                for j in range(i, len(df)):
                    if abs(df['Roll_Smooth'].iloc[j] - current_hump['peak_value']) <= STABILITY_TOLERANCE:
                        is_stable = True
                        if j + MIN_STABLE_FRAMES >= len(df): is_stable = False; break
                        for k in range(1, MIN_STABLE_FRAMES + 1):
                            if abs(df['Roll_Smooth'].iloc[j+k] - current_hump['peak_value']) > STABILITY_TOLERANCE: is_stable = False; break
                        if is_stable: current_hump['stabilization_start_frame'] = df['Frame'].iloc[j]; break
                if current_hump['stabilization_start_frame'] == -1: current_hump['stabilization_start_frame'] = current_hump['peak_frame']
        elif state == "PEAK_STABILIZING":
            if curr_val < current_hump['peak_value'] - RISE_THRESHOLD and curr_val < prev_val:
                state = "FALLING"; current_hump['descent_start_frame'] = df['Frame'].iloc[i]
        elif state == "FALLING":
            if curr_val <= baseline + STABILITY_TOLERANCE:
                state = "IDLE"
                current_hump['end_frame'] = df['Frame'].iloc[i]
                all_humps.append(current_hump)
                current_hump = {}

    filtered_humps = []
    min_duration_frames = MIN_PEAK_TO_END_DURATION_SECONDS * SAMPLING_RATE
    for i, hump in enumerate(all_humps):
        peak_to_end_duration_frames = hump.get('end_frame', 0) - hump.get('peak_frame', 0)
        
        if peak_to_end_duration_frames >= min_duration_frames:
            filtered_humps.append(hump)
        else:
            logging.warning(f"구간 제외: Hump #{i+1}의 Peak-to-End 소요 시간이 {peak_to_end_duration_frames / SAMPLING_RATE:.2f}초로, 설정된 최소 시간({MIN_PEAK_TO_END_DURATION_SECONDS}초)보다 짧습니다.")

    create_debug_plot(df, baseline, rise_threshold_value, filtered_humps, file_path)

    if not filtered_humps:
        logging.warning(f"분석 완료: '{file_path.name}'에서 유의미한 움직임 구간을 찾지 못했습니다. 파라미터를 조정해보세요."); return

    analysis_results = []
    for i, hump in enumerate(filtered_humps):
        start_time = hump.get('start_frame', 0) / SAMPLING_RATE
        peak_time = hump.get('peak_frame', 0) / SAMPLING_RATE
        stabilization_start_time = hump.get('stabilization_start_frame', 0) / SAMPLING_RATE
        descent_start_time = hump.get('descent_start_frame', 0) / SAMPLING_RATE
        end_time = hump.get('end_frame', 0) / SAMPLING_RATE
        analysis_results.append({
            'Hump_Index': i + 1,
            'Start_Time(s)': start_time,
            'Peak_Time(s)': peak_time,
            'Start_Duration(s)': peak_time - start_time,
            'Stabilization_Time(s)': stabilization_start_time,
            'Stabilization_Duration(s)': descent_start_time - stabilization_start_time,
            'Descent_Start_Time(s)': descent_start_time,
            'End_Time(s)': end_time,
            'Descent_Duration(s)': end_time - descent_start_time,
            'Peak_Roll_Value(deg)': hump.get('peak_value', 0)
        })

    analysis_df = pd.DataFrame(analysis_results)
    summary = pd.Series({'Total_Humps': len(filtered_humps)}, name='Summary')
    analysis_df = pd.concat([analysis_df, summary.to_frame().T], ignore_index=False)
    output_filename = output_dir / f"{file_path.stem}_analysis.csv"
    analysis_df.to_csv(output_filename, index=False, float_format='%.3f')
    logging.info(f"분석 완료: 결과가 '{output_filename}'에 저장되었습니다.")


if __name__ == '__main__':
    try:
        input_path = Path(TARGET_DIRECTORY)
        if not input_path.is_dir():
            if "여기에 분석할 폴더의 전체 경로를 붙여넣으세요" in TARGET_DIRECTORY:
                 print("\n[오류] 스크립트의 TARGET_DIRECTORY 변수에 분석할 폴더 경로를 먼저 입력해주세요.")
            else: logging.error(f"설정된 경로가 폴더가 아니거나 존재하지 않습니다: {input_path}")
            sys.exit(1)

        for csv_file in sorted(input_path.glob('*.csv')):
            if csv_file.name == 'markers.csv' or '_analysis.csv' in csv_file.name: continue
            logging.info(f"--- '{csv_file.name}' 파일 분석 시작 ---")
            analyze_roll_data(csv_file, input_path)
        
        logging.info("모든 파일 분석이 완료되었습니다.")
    except Exception as e:
        logging.error(f"예상치 못한 오류가 발생했습니다: {e}")
        sys.exit(1)
