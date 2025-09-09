"""
Stream Rigid-Bodies + Labeled-Markers ▶ CSV 저장 (장시간 안정 버전)
────────────────────────────────────────────────
• 실시간 스트리밍 (Capture 없이)
• 주기적으로 데이터를 파일에 저장하여 메모리 문제 방지
• CSV 파일의 Frame 열이 1부터 시작 (QTM 리셋 감지)
• Ctrl + C  → 남은 데이터 저장 후 종료 (종료 안정성 강화)
"""

import asyncio
import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
import pandas as pd
from scipy.spatial.transform import Rotation as R
import qtm_rt as qtm

# ────────── 사용자 설정 ──────────
# QTM 연결 정보
QTM_IP       = "127.0.0.1"
QTM_PASSWORD = "password"

# 데이터 저장 설정
SAVE_INTERVAL_SECONDS = 600
WANTED_BODY  = None

# QTM 파일 재생 모드 (실시간 스트리밍 시에는 None으로 설정)
QTM_FILE     = None
# ──────────────────────

# 로깅 설정
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

# 전역 변수 및 초기화
STAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTDIR  = Path.home() / f"desktop/qtm_export_{STAMP}"
OUTDIR.mkdir(parents=True, exist_ok=True)
START_TIME = datetime.now()

# 데이터 임시 저장소 (메모리 버퍼)
body_rows: dict[str, list[list]] = {}
marker_rows: dict[str, list[list]] = {}

# ── 헬퍼 함수 ──
def body_index(xml_str: str) -> dict[str, int]:
    """XML 데이터에서 바디 이름과 인덱스를 추출합니다."""
    root = ET.fromstring(xml_str)
    return {e.text.strip(): i for i, e in enumerate(root.findall("*/Body/Name"))}

def rotation_to_rpy(matrix: list[float]) -> tuple[float, float, float]:
    """3×3 회전 행렬을 Roll/Pitch/Yaw (deg)로 변환합니다."""
    try:
        r = R.from_matrix([[matrix[0], matrix[1], matrix[2]],
                           [matrix[3], matrix[4], matrix[5]],
                           [matrix[6], matrix[7], matrix[8]]])
        return r.as_euler("xyz", degrees=True)
    except (ValueError, IndexError):
        return float("nan"), float("nan"), float("nan")

# ── 데이터 저장 함수 ──
def flush_data_to_disk():
    """메모리에 쌓인 데이터를 CSV 파일에 추가로 기록하고 버퍼를 비웁니다."""
    global body_rows, marker_rows
    if not body_rows and not marker_rows:
        return

    LOG.info(f"[{datetime.now().strftime('%H:%M:%S')}] 데이터를 파일에 저장합니다...")

    header_b = ["Frame", "Pos_X", "Pos_Y", "Pos_Z", "Roll", "Pitch", "Yaw"]
    for name, rows in body_rows.items():
        if not rows: continue
        path = OUTDIR / f"{name}.csv"
        is_new_file = not path.exists()
        df = pd.DataFrame(rows, columns=header_b)
        df.to_csv(path, mode='a', header=is_new_file, index=False)

    if any(marker_rows.values()):
        temp_dfs = []
        for key, rows in marker_rows.items():
            if not rows: continue
            df = pd.DataFrame(rows, columns=["Frame", f"{key}_Pos_X", f"{key}_Pos_Y", f"{key}_Pos_Z"])
            df = df.set_index("Frame")
            temp_dfs.append(df)
        
        if temp_dfs:
            merged_df = pd.concat(temp_dfs, axis=1)
            path = OUTDIR / "markers.csv"
            is_new_file = not path.exists()
            merged_df.to_csv(path, mode='a', header=is_new_file, index=True)

    body_rows = {name: [] for name in body_rows}
    marker_rows = {name: [] for name in marker_rows}
    LOG.info("저장 완료. 메모리 버퍼를 비웠습니다.")

# ── 주기적 저장 태스크 ──
async def periodic_saver():
    """설정된 시간 간격마다 데이터 저장 함수를 호출하는 비동기 태스크."""
    while True:
        await asyncio.sleep(SAVE_INTERVAL_SECONDS)
        flush_data_to_disk()

# ── 메인 로직 ──
async def main():
    """QTM에 연결하고 데이터 스트리밍 및 저장을 관리합니다."""
    conn = await qtm.connect(QTM_IP)
    if conn is None:
        LOG.error("❌ QTM 연결 실패"); return

    saver_task = asyncio.create_task(periodic_saver())
    
    frame_offset = None
    last_framenumber = -1

    try:
        await conn.take_control(QTM_PASSWORD)
        LOG.info("QTM 제어권을 획득했습니다.")

        if QTM_FILE:
            await conn.load(QTM_FILE)
            await conn.start(rtfromfile=True)
            LOG.info(f"▶ '{QTM_FILE}' 파일 재생 중…")
        else:
            LOG.info("📡 실시간 스트리밍 시작… (Capture 없이)")

        bmap = body_index(await conn.get_parameters(["6d"]))
        xml3d = await conn.get_parameters(["3d"])
        m_name_list = [e.text.strip() for e in ET.fromstring(xml3d).findall("*/Marker/Name")]

        def on_packet(pkt: qtm.QRTPacket):
            """수신된 각 데이터 패킷을 처리하여 메모리 버퍼에 추가합니다."""
            nonlocal frame_offset, last_framenumber
            
            if last_framenumber != -1 and pkt.framenumber < last_framenumber - 100:
                LOG.warning(f"QTM 프레임 카운터 리셋 감지: {last_framenumber} -> {pkt.framenumber}. 기준점을 재설정합니다.")
                frame_offset = None

            if frame_offset is None:
                frame_offset = pkt.framenumber
            
            relative_fnum = pkt.framenumber - frame_offset + 1
            last_framenumber = pkt.framenumber

            _, bodies = pkt.get_6d()
            _, markers = pkt.get_3d_markers()

            if bodies:
                for name, idx in bmap.items():
                    if WANTED_BODY and name != WANTED_BODY: continue
                    if idx < len(bodies) and bodies[idx]:
                        pos, rot = bodies[idx]
                        roll, pitch, yaw = rotation_to_rpy(rot.matrix)
                        body_rows.setdefault(name, []).append(
                            [relative_fnum, pos.x, pos.y, pos.z, roll, pitch, yaw]
                        )
            if markers:
                for i, m in enumerate(markers):
                    if m is None: continue
                    key = m_name_list[i] if i < len(m_name_list) else f"M{i}"
                    marker_rows.setdefault(key, []).append([relative_fnum, m.x, m.y, m.z])

        await conn.stream_frames(components=["6d", "3d"], on_packet=on_packet)
        LOG.info("데이터 수신을 시작합니다. 종료하려면 Ctrl+C를 누르세요.")
        await asyncio.Future()

    except asyncio.CancelledError:
        LOG.info("프로그램이 외부 신호에 의해 종료됩니다.")
    except Exception as e:
        LOG.error(f"오류 발생: {e}", exc_info=True)
    finally:
        saver_task.cancel()
        if conn and conn.has_transport():
            await conn.stream_frames_stop()
            try:
                # [수정] 제어권 해제 시 타임아웃 오류 방지
                await conn.release_control()
                LOG.info("QTM 제어권을 해제했습니다.")
            except asyncio.TimeoutError:
                LOG.warning("QTM 제어권 해제 실패 (타임아웃). 연결이 비정상적일 수 있습니다.")
            conn.disconnect()
            
        LOG.info("\n🛑 스트리밍 종료. 남은 데이터를 최종 저장합니다...")
        flush_data_to_disk()
        LOG.info(f"✅ 모든 데이터 저장 완료 → {OUTDIR}")

# ── 실행 ──
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        LOG.info("\n프로그램을 종료합니다.")
