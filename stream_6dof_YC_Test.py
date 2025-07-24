"""
Stream Rigid-Bodies + Labeled-Markers ▶ CSV 저장
────────────────────────────────────────────────
• 실시간 스트리밍 (Capture 없이)
• 's' 키 입력 → 실시간 데이터 종료 후 → CSV 저장 (시작/종료 시간 포함)
"""

import asyncio
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
import pandas as pd
from scipy.spatial.transform import Rotation as R
import qtm_rt as qtm
import keyboard  # 's' 키 입력을 감지하기 위해 추가

# ────────── 사용자 설정 ──────────
QTM_FILE = r"C:\Users\user\Documents\Qualisys\Data\AIM\RigidBody_Vcam_V01.qtm"  # r"C:\QTM\YourFile.qtm" 또는 None → 라이브
QTM_IP = "127.0.0.1"
QTM_PASSWORD = "password"
WANTED_BODY = None  # 특정 바디만: "Bat", 전부 저장하려면 None
# ──────────────────────

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTDIR = Path.home() / f"Desktop/qtm_export_{STAMP}"
OUTDIR.mkdir(parents=True, exist_ok=True)
START_TIME = datetime.now()

body_rows: dict[str, list[list]] = {}
marker_rows: dict[str, list[list]] = {}

# ── 헬퍼 ──
def body_index(xml_str: str) -> dict[str, int]:
    """XML 문자열에서 Rigid Body의 이름과 인덱스를 매핑합니다."""
    root = ET.fromstring(xml_str)
    return {e.text.strip(): i for i, e in enumerate(root.findall("*/Body/Name"))}

def rotation_to_rpy(matrix: list[float]) -> tuple[float, float, float]:
    """3×3 회전 행렬 → Roll/Pitch/Yaw (deg). 실패 시 NaN 반환"""
    try:
        r = R.from_matrix([[matrix[0], matrix[1], matrix[2]],
                           [matrix[3], matrix[4], matrix[5]],
                           [matrix[6], matrix[7], matrix[8]]])
        return r.as_euler("xyz", degrees=True)
    except Exception:
        return float("nan"), float("nan"), float("nan")

# ── 메인 ──
async def main():
    """QTM에 연결하고 데이터 스트리밍을 시작합니다."""
    conn = await qtm.connect(QTM_IP)
    if conn is None:
        print("❌ QTM 연결 실패")
        return

    # 종료 신호를 위한 asyncio.Event 생성
    stop_event = asyncio.Event()

    def signal_stop():
        """'s' 키가 눌렸을 때 호출되어 종료 이벤트를 설정하는 콜백 함수"""
        if not stop_event.is_set():
            print("\n🛑 's' 키 입력 감지. 종료 및 저장을 시작합니다...")
            stop_event.set()

    # 's' 키가 눌리면 signal_stop 함수를 호출하도록 등록
    keyboard.on_press_key("s", lambda _: signal_stop())

    try:
        # QTM 연결 및 실시간 스트리밍 시작 (Capture 없이)
        async with qtm.TakeControl(conn, QTM_PASSWORD):
            if QTM_FILE:
                await conn.load(QTM_FILE)
                await conn.start(rtfromfile=True)
                print("▶ 측정 파일 재생 중…")
            else:  # 실시간 스트리밍 (Capture 없이)
                await conn.new()
                await conn.start(rtfromfile=False)
                print("📡 실시간 스트리밍 시작…")

            print("\n데이터 수집 중입니다. 중지하고 저장하려면 's' 키를 누르세요.")

            bmap = body_index(await conn.get_parameters(["6d"]))
            xml3d = await conn.get_parameters(["3d"])
            m_name_list = [e.text.strip() for e in ET.fromstring(xml3d).findall("*/Marker/Name")]

            # ─ 패킷 콜백 ─
            def on_packet(pkt):
                """수신된 각 데이터 패킷을 처리합니다."""
                fnum = pkt.framenumber
                _, bodies = pkt.get_6d()
                _, markers = pkt.get_3d_markers()

                # 리지드 바디 저장
                for name, idx in bmap.items():
                    if WANTED_BODY and name != WANTED_BODY: continue
                    pos, rot = bodies[idx]
                    roll, pitch, yaw = rotation_to_rpy(rot.matrix)
                    body_rows.setdefault(name, []).append(
                        [fnum, pos.x, pos.y, pos.z, roll, pitch, yaw]
                    )

                # 마커 저장
                for i, m in enumerate(markers):
                    if m is None: continue
                    key = m_name_list[i] if i < len(m_name_list) else f"M{i}"
                    x, y, z = m
                    marker_rows.setdefault(key, []).append([fnum, x, y, z])

            # 스트리밍 시작
            await conn.stream_frames(components=["6d", "3d"], on_packet=on_packet)

            # stop_event가 설정될 때까지 대기 ('s' 키를 누를 때까지)
            await stop_event.wait()

            # 스트리밍 중지
            await conn.stream_frames_stop()
            try:
                await conn.stop()
            except Exception:
                pass

    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        # 키보드 리스너 해제
        keyboard.unhook_all()
        # 데이터 저장
        save_data()
        # QTM 연결 해제
        conn.disconnect()

# ── 저장 ──
def save_data():
    """수집된 데이터를 CSV 파일로 저장합니다."""
    END_TIME = datetime.now()
    if not body_rows and not marker_rows:
        print("⚠️ 수집된 데이터가 없습니다.")
        return

    print("💾 데이터를 CSV 파일로 저장 중…")
    head_time = f"# Start: {START_TIME}\n# End:   {END_TIME}\n"

    # 리지드 바디 개별 저장
    header_b = ["Frame", "Pos_X", "Pos_Y", "Pos_Z", "Roll", "Pitch", "Yaw"]
    for name, rows in body_rows.items():
        path = OUTDIR / f"{name}.csv"
        df = pd.DataFrame(rows, columns=header_b)
        
        # 프레임 번호를 기준으로 중복 제거 및 보간
        if not df.empty:
            df = df.drop_duplicates(subset=["Frame"], keep="last")
            df = df.set_index("Frame")
            full_frame_range = range(int(df.index.min()), int(df.index.max()) + 1)
            df = df.reindex(full_frame_range).interpolate(method='linear')
        
        with path.open("w", newline="", encoding='utf-8') as f:
            f.write(head_time)
            df.to_csv(f, index=True)

    # 마커 통합 저장
    if marker_rows:
        all_frames = set()
        for rows in marker_rows.values():
            for row in rows:
                all_frames.add(row[0])
        
        if all_frames:
            sorted_frames = sorted(list(all_frames))
            
            # 모든 마커 데이터를 포함할 마스터 DataFrame 생성
            master_df = pd.DataFrame(index=pd.Index(sorted_frames, name="Frame"))

            for key, rows in marker_rows.items():
                df = pd.DataFrame(rows, columns=["Frame", f"{key}_X", f"{key}_Y", f"{key}_Z"])
                df = df.drop_duplicates(subset=["Frame"], keep='last').set_index("Frame")
                master_df = master_df.join(df, how='left')

            # 전체 데이터에 대해 보간
            master_df = master_df.interpolate(method='linear')
            
            path = OUTDIR / "markers.csv"
            with path.open("w", newline="", encoding='utf-8') as f:
                f.write(head_time)
                master_df.to_csv(f, index=True)

    print(f"✅ 저장 완료 → {OUTDIR}")

# ── 실행 ─
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n프로그램 강제 종료.")
