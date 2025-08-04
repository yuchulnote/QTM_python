"""
Stream Rigid-Bodies + Labeled-Markers ▶ CSV 저장 (최종 기능 개선 버전)
────────────────────────────────────────────────
• 실시간 스트리밍 (Capture 없이)
• 주기적으로 데이터를 파일에 저장하여 메모리 문제 방지
• CSV 파일의 Frame 열이 1부터 시작 (QTM 리셋 감지)
• 시작/종료 시간 메타데이터 파일에 기록
• 저장 경로 사용자 설정 가능
• QTM 연결 종료 시 자동 종료 및 저장
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
# [추가] 데이터가 저장될 폴더 경로 (예: "D:/QTM_Data" 또는 str(Path.home() / "Documents"))
SAVE_PATH    = str(Path.home() / "desktop")


# QTM 파일 재생 모드 (실시간 스트리밍 시에는 None으로 설정)
QTM_FILE     = None
# ──────────────────────

# 로깅 설정
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

# 전역 변수 및 초기화
STAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTDIR  = Path(SAVE_PATH) / f"qtm_export_{STAMP}"
OUTDIR.mkdir(parents=True, exist_ok=True)
START_TIME = datetime.now()

# 데이터 임시 저장소 (메모리 버퍼)
body_rows: dict[str, list[list]] = {}
marker_rows: dict[str, list[list]] = {}

# ── 헬퍼 함수 ──
def body_index(xml_str: str) -> dict[str, int]:
    root = ET.fromstring(xml_str)
    return {e.text.strip(): i for i, e in enumerate(root.findall("*/Body/Name"))}

def rotation_to_rpy(matrix: list[float]) -> tuple[float, float, float]:
    try:
        r = R.from_matrix([[matrix[0], matrix[1], matrix[2]],
                           [matrix[3], matrix[4], matrix[5]],
                           [matrix[6], matrix[7], matrix[8]]])
        return r.as_euler("xyz", degrees=True)
    except (ValueError, IndexError):
        return float("nan"), float("nan"), float("nan")

# ── 데이터 저장 함수 ──
def flush_data_to_disk():
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

    body_rows.clear()
    marker_rows.clear()
    LOG.info("저장 완료. 메모리 버퍼를 비웠습니다.")

# ── 주기적 저장 태스크 ──
async def periodic_saver():
    while True:
        await asyncio.sleep(SAVE_INTERVAL_SECONDS)
        flush_data_to_disk()

# ── 메인 로직 ──
async def main():
    # [추가] 시작 시간 메타데이터 파일에 기록
    with open(OUTDIR / "_metadata.txt", "w", encoding="utf-8") as f:
        f.write(f"Recording Start Time: {START_TIME.isoformat()}\n")

    # [추가] 연결 끊김 감지 시 메인 태스크를 취소하는 콜백
    main_task = asyncio.current_task()
    def on_disconnect(reason):
        LOG.warning(f"QTM 연결이 끊어졌습니다 (Reason: {reason}). 5초 후 프로그램을 종료합니다.")
        if not main_task.done():
            # 즉시 종료하면 최종 데이터 저장이 안될 수 있으므로 약간의 지연 후 취소
            asyncio.get_event_loop().call_later(5, main_task.cancel)

    conn = await qtm.connect(QTM_IP, on_disconnect=on_disconnect)
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
        LOG.info("프로그램 종료 신호를 받았습니다.")
    except Exception as e:
        LOG.error(f"오류 발생: {e}", exc_info=True)
    finally:
        saver_task.cancel()
        if conn and conn.has_transport():
            await conn.stream_frames_stop()
            try:
                await conn.release_control()
                LOG.info("QTM 제어권을 해제했습니다.")
            except asyncio.TimeoutError:
                LOG.warning("QTM 제어권 해제 실패 (타임아웃).")
            conn.disconnect()
            
        LOG.info("\n🛑 스트리밍 종료. 남은 데이터를 최종 저장합니다...")
        flush_data_to_disk()

        # [추가] 종료 시간 메타데이터 파일에 기록
        with open(OUTDIR / "_metadata.txt", "a", encoding="utf-8") as f:
            f.write(f"Recording End Time:   {datetime.now().isoformat()}\n")

        LOG.info(f"✅ 모든 데이터 저장 완료 → {OUTDIR}")

# ── 실행 ──
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        LOG.info("\n프로그램을 종료합니다.")
