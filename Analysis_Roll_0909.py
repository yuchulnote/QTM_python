import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# --- 분석 파라미터 설정 (이곳에서 모든 값을 조정하세요) ---
TARGET_DIRECTORY = r"C:\Users\user\Desktop\real-Time\QTM_python\qtm_export_20250825_094548"
SAMPLING_RATE = 120.0
SMOOTHING_WINDOW = 60
RISE_THRESHOLD = 2.0
STABILITY_TOLERANCE = 2.0
MIN_STABLE_FRAMES = int(SAMPLING_RATE * 2.0) # 0.5초
BASELINE_ZERO_TOLERANCE = 2.0
MIN_PEAK_TO_END_DURATION_SECONDS = 20.0
# --------------------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def combine_plots(plot_paths: list, output_dir: Path):
    """여러 플롯 이미지를 하나의 세로 이미지로 결합합니다."""
    if not plot_paths:
        logging.info("결합할 플롯이 없습니다.")
        return

    logging.info(f"{len(plot_paths)}개의 플롯을 하나의 이미지로 결합합니다...")
    
    plot_paths.sort()

    try:
        images = [mpimg.imread(path) for path in plot_paths]
    except FileNotFoundError as e:
        logging.error(f"플롯 이미지를 읽는 중 오류 발생: {e}. 요약 이미지를 생성할 수 없습니다.")
        return

    fig, axs = plt.subplots(len(images), 1, figsize=(20, 10 * len(images)))
    
    if len(images) == 1:
        axs = [axs]

    for ax, img, path in zip(axs, images, plot_paths):
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(Path(path).name, fontsize=14, pad=10)

    plt.tight_layout()
    summary_plot_path = output_dir / "_All_Plots_Summary.png"
    plt.savefig(summary_plot_path)
    plt.close()
    logging.info(f"모든 플롯을 결합한 요약 이미지가 '{summary_plot_path}'에 저장되었습니다.")


def create_debug_plot(df: pd.DataFrame, humps: list, baselines: list, output_path: Path) -> Path:
    plt.figure(figsize=(20, 8))
    plt.plot(df['Time'], df['Roll'], color='lightgray', label='Original Roll')
    plt.plot(df['Time'], df['Roll_Smooth'], color='blue', label='Smoothed Roll')

    for i, b in enumerate(baselines):
        plt.hlines(y=b['value'], xmin=b['start_time'], xmax=b['end_time'], color='orange', linestyle='--', label=f'Baseline {i+1}' if i==0 else "")
        plt.hlines(y=b['value'] + RISE_THRESHOLD, xmin=b['start_time'], xmax=b['end_time'], color='red', linestyle=':', label='Threshold' if i==0 else "")

    for i, hump in enumerate(humps):
        start_time = hump.get('start_frame', 0) / SAMPLING_RATE
        peak_time = hump.get('peak_frame', 0) / SAMPLING_RATE
        end_time = hump.get('end_frame', 0) / SAMPLING_RATE
        plt.axvline(x=start_time, color='green', linestyle=':', label=f'Hump {i+1} Start' if i==0 else "")
        plt.axvline(x=peak_time, color='purple', linestyle=':', label=f'Hump {i+1} Peak' if i==0 else "")
        plt.axvline(x=end_time, color='black', linestyle=':', label=f'Hump {i+1} End' if i==0 else "")

    plt.title(f'Roll Analysis Debug Plot - {output_path.stem}')
    plt.xlabel('Time (s)'); plt.ylabel('Roll (degrees)')
    plt.legend(); plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plot_filename = output_path.parent / f"{output_path.stem}_plot.png"
    plt.savefig(plot_filename)
    plt.close()
    logging.info(f"플롯이 '{plot_filename}'에 저장되었습니다.")
    return plot_filename

def analyze_roll_data(file_path: Path, output_dir: Path) -> dict:
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logging.error(f"파일을 찾을 수 없습니다: {file_path}"); return None

    df['Roll'] = df['Roll'] * -1
    if 'Roll' not in df.columns or 'Frame' not in df.columns:
        logging.warning(f"'{file_path.name}' 파일에 'Roll' 또는 'Frame' 열이 없어 건너뜁니다."); return None

    df['Roll_Smooth'] = df['Roll'].rolling(window=SMOOTHING_WINDOW, min_periods=1, center=True).mean()
    df['Time'] = df['Frame'] / SAMPLING_RATE

    baseline = None; baseline_start_frame = 0
    for i in range(len(df) - MIN_STABLE_FRAMES):
        window = df['Roll_Smooth'].iloc[i : i + MIN_STABLE_FRAMES]
        if (window.abs() < BASELINE_ZERO_TOLERANCE).all():
            baseline = window.mean()
            baseline_start_frame = i
            logging.info(f"초기 기준선 탐색 성공. {i/SAMPLING_RATE:.2f}초 부근에서 기준선을 {baseline:.2f}(으)로 설정합니다.")
            break
    if baseline is None:
        baseline = df['Roll_Smooth'].iloc[:int(SAMPLING_RATE)].mean()
        logging.warning(f"안정적인 0도 부근을 찾지 못했습니다. 파일 시작 1초를 기준으로 강제 설정합니다 (기준선: {baseline:.2f}).")

    state = "IDLE"; all_humps = []; current_hump = {}; baselines_for_plot = []
    i = baseline_start_frame
    
    while i < len(df) - 1:
        rise_threshold_value = baseline + RISE_THRESHOLD
        prev_val, curr_val = df['Roll_Smooth'].iloc[i], df['Roll_Smooth'].iloc[i+1]
        
        if state == "IDLE":
            if curr_val > baseline + STABILITY_TOLERANCE and curr_val > prev_val:
                state = "PREP_RISE"; current_hump = {'prep_start_frame': df['Frame'].iloc[i+1]}
        elif state == "PREP_RISE":
            if curr_val > rise_threshold_value:
                true_start_index = i
                for j in range(i, baseline_start_frame, -1):
                    if df['Roll_Smooth'].iloc[j] <= baseline + STABILITY_TOLERANCE:
                        true_start_index = j + 1; break
                state = "RISING"
                current_hump['start_frame'] = df['Frame'].iloc[true_start_index]
                baselines_for_plot.append({'value': baseline, 'start_time': df['Time'].iloc[true_start_index], 'end_time': -1})
            elif curr_val < prev_val and curr_val < baseline + STABILITY_TOLERANCE:
                state = "IDLE"; current_hump = {}
        elif state == "RISING":
            if curr_val <= prev_val:
                state = "PEAK_STABILIZING"
                current_hump.update({'peak_frame': df['Frame'].iloc[i], 'peak_value': prev_val, 'stabilization_start_frame': -1})
                for j in range(i + 1, len(df)):
                    if abs(df['Roll_Smooth'].iloc[j] - current_hump['peak_value']) <= STABILITY_TOLERANCE:
                        is_stable = True
                        if j + MIN_STABLE_FRAMES >= len(df): is_stable = False; break
                        for k in range(1, MIN_STABLE_FRAMES + 1):
                            if abs(df['Roll_Smooth'].iloc[j+k] - current_hump['peak_value']) > STABILITY_TOLERANCE: is_stable = False; break
                        if is_stable: current_hump['stabilization_start_frame'] = df['Frame'].iloc[j]; break
                if current_hump['stabilization_start_frame'] == -1: current_hump['stabilization_start_frame'] = current_hump['peak_frame']
        elif state == "PEAK_STABILIZING":
            if curr_val < current_hump['peak_value'] - STABILITY_TOLERANCE and curr_val < prev_val:
                state = "FALLING"; current_hump['descent_start_frame'] = df['Frame'].iloc[i+1]
        elif state == "FALLING":
            if curr_val < rise_threshold_value:
                is_now_stable = False
                if i + MIN_STABLE_FRAMES < len(df):
                    window = df['Roll_Smooth'].iloc[i : i + MIN_STABLE_FRAMES]
                    if window.max() - window.min() < STABILITY_TOLERANCE: is_now_stable = True
                if is_now_stable:
                    state = "IDLE"
                    new_baseline = window.mean()
                    current_hump['end_frame'] = df['Frame'].iloc[i]
                    current_hump['new_baseline'] = new_baseline
                    all_humps.append(current_hump)
                    logging.info(f"새로운 기준선 발견. {df['Time'].iloc[i]:.2f}초 부근에서 기준선을 {new_baseline:.2f}(으)로 업데이트합니다.")
                    if baselines_for_plot: baselines_for_plot[-1]['end_time'] = df['Time'].iloc[i]
                    baseline = new_baseline; baseline_start_frame = i; current_hump = {}
                    i += MIN_STABLE_FRAMES
                    continue
        i += 1

    if baselines_for_plot: baselines_for_plot[-1]['end_time'] = df['Time'].iloc[-1]

    filtered_humps = []
    min_duration_frames = MIN_PEAK_TO_END_DURATION_SECONDS * SAMPLING_RATE
    for idx, hump in enumerate(all_humps):
        peak_to_end_duration_frames = hump.get('end_frame', 0) - hump.get('peak_frame', 0)
        if peak_to_end_duration_frames >= min_duration_frames:
            filtered_humps.append(hump)
        else:
            logging.warning(f"구간 제외: Hump #{idx+1}의 Peak-to-End 소요 시간이 {peak_to_end_duration_frames / SAMPLING_RATE:.2f}초로, 설정된 최소 시간({MIN_PEAK_TO_END_DURATION_SECONDS}초)보다 짧습니다.")

    plot_filename = create_debug_plot(df, filtered_humps, baselines_for_plot, file_path)
    analysis_df = pd.DataFrame()

    if not filtered_humps:
        logging.warning(f"분석 완료: '{file_path.name}'에서 유의미한 움직임 구간을 찾지 못했습니다.")
    else:
        analysis_results = []
        prev_baseline = baselines_for_plot[0]['value'] if baselines_for_plot else 0
        for i, hump in enumerate(filtered_humps):
            new_baseline = hump.get('new_baseline')
            baseline_shifted = new_baseline is not None and not np.isclose(new_baseline, prev_baseline)
            stabilization_start_frame = hump.get('stabilization_start_frame', 0)
            descent_start_frame = hump.get('descent_start_frame', 0)
            avg_stabilization_roll = np.nan
            if stabilization_start_frame > 0 and descent_start_frame > stabilization_start_frame:
                try:
                    start_idx = df.index[df['Frame'] == stabilization_start_frame].tolist()[0]
                    end_idx = df.index[df['Frame'] == descent_start_frame].tolist()[0]
                    stabilization_data = df['Roll_Smooth'].iloc[start_idx:end_idx]
                    if not stabilization_data.empty: avg_stabilization_roll = stabilization_data.mean()
                except IndexError: logging.warning(f"Hump #{i+1}: Stabilization 구간의 프레임 인덱스를 찾지 못했습니다.")
            if pd.isna(avg_stabilization_roll): avg_stabilization_roll = hump.get('peak_value', 0)
            analysis_results.append({
                'Hump_Index': i + 1, 'Start_Time(s)': hump.get('start_frame', 0) / SAMPLING_RATE,
                'Peak_Time(s)': hump.get('peak_frame', 0) / SAMPLING_RATE,
                'Start_duration(s)': (hump.get('peak_frame', 0) - hump.get('start_frame', 0)) / SAMPLING_RATE,
                'Stabilization_Time(s)': hump.get('stabilization_start_frame', 0) / SAMPLING_RATE,
                'Stabilization_Duration(s)': (hump.get('stabilization_start_frame', 0) - hump.get('start_frame', 0)) / SAMPLING_RATE,
                'Descent_Start_Time(s)': hump.get('descent_start_frame', 0) / SAMPLING_RATE,
                'End_Time(s)': hump.get('end_frame', 0) / SAMPLING_RATE,
                'Descent_Duration(s)': (hump.get('end_frame', 0) - hump.get('descent_start_frame', 0)) / SAMPLING_RATE,
                'Avg_Stabilization_Roll_Value(deg)': avg_stabilization_roll,
                'New_Baseline_Value(deg)': new_baseline if baseline_shifted else np.nan,
                'Caused_Baseline_Shift': baseline_shifted
            })
            if baseline_shifted: prev_baseline = new_baseline
        analysis_df = pd.DataFrame(analysis_results)
        output_filename = output_dir / f"{file_path.stem}_analysis.csv"
        analysis_df.to_csv(output_filename, index=False, float_format='%.3f')
        logging.info(f"분석 완료: 결과가 '{output_filename}'에 저장되었습니다.")
    
    return {'plot_path': plot_filename, 'analysis_df': analysis_df}

if __name__ == '__main__':
    try:
        input_path = Path(TARGET_DIRECTORY)
        if not input_path.is_dir():
            if "여기에 분석할 폴더의 전체 경로를 붙여넣으세요" in TARGET_DIRECTORY:
                 print("\n[오류] 스크립트의 TARGET_DIRECTORY 변수에 분석할 폴더 경로를 먼저 입력해주세요.")
            else: logging.error(f"설정된 경로가 폴더가 아니거나 존재하지 않습니다: {input_path}")
            sys.exit(1)
        
        generated_plots = []
        chair_analysis_data = {}

        files_to_process = sorted([f for f in input_path.glob('*.csv') if f.name != 'markers.csv' and '_analysis.csv' not in f.name and '_Chair' not in f.name])
        
        if not files_to_process:
            logging.warning(f"'{input_path}' 폴더에서 분석할 CSV 파일을 찾지 못했습니다.")
        else:
            for csv_file in files_to_process:
                logging.info(f"--- '{csv_file.name}' 파일 분석 시작 ---")
                
                parts = csv_file.stem.split('_')
                if len(parts) < 2:
                    logging.warning(f"'{csv_file.name}' 파일 이름 형식이 올바르지 않아 건너뜁니다 (예: 1_TOP.csv).")
                    continue
                
                chair_id, body_part = parts[0], parts[1]
                result = analyze_roll_data(csv_file, input_path)

                if result:
                    if result.get('plot_path'):
                        generated_plots.append(result['plot_path'])
                    if not result['analysis_df'].empty:
                        if chair_id not in chair_analysis_data:
                            chair_analysis_data[chair_id] = {}
                        chair_analysis_data[chair_id][body_part] = result['analysis_df']
            
            # --- 모든 파일 분석 후 의자별/피크별 요약 분석 ---
            logging.info("--- 의자별 피크 이벤트 분석 시작 ---")
            for chair_id, parts_data in chair_analysis_data.items():
                max_peaks = 0
                for df in parts_data.values():
                    if len(df) > max_peaks:
                        max_peaks = len(df)
                
                if max_peaks == 0:
                    logging.warning(f"Chair '{chair_id}': 분석할 피크가 없습니다.")
                    continue

                chair_results = []
                for i in range(max_peaks): # i는 0부터 시작하는 피크 인덱스
                    earliest_start = float('inf')
                    latest_end = float('-inf')
                    
                    # 이 피크를 구성하는 파트들의 데이터를 수집
                    for part_df in parts_data.values():
                        if i < len(part_df): # 해당 파트에 i번째 피크가 존재하는 경우
                            start_time = part_df.iloc[i]['Start_Time(s)']
                            end_time = part_df.iloc[i]['End_Time(s)']
                            if start_time < earliest_start: earliest_start = start_time
                            if end_time > latest_end: latest_end = end_time
                    
                    if np.isfinite(earliest_start) and np.isfinite(latest_end):
                        duration = latest_end - earliest_start
                        chair_results.append({
                            'Peak_Index': i + 1,
                            'Earliest_Start_Time(s)': earliest_start,
                            'Latest_End_Time(s)': latest_end,
                            'Operating_Duration(s)': duration
                        })

                if chair_results:
                    summary_df = pd.DataFrame(chair_results)
                    output_filename = input_path / f"{chair_id}_Chair_analysis.csv"
                    summary_df.to_csv(output_filename, index=False, float_format='%.3f')
                    logging.info(f"Chair '{chair_id}' 요약 분석 완료 → '{output_filename}'")

            # 생성된 모든 플롯을 하나의 이미지로 결합
            if generated_plots:
                combine_plots(generated_plots, input_path)

        logging.info("모든 파일 분석이 완료되었습니다.")
    except Exception as e:
        logging.error(f"예상치 못한 오류가 발생했습니다: {e}", exc_info=True)
        sys.exit(1)