"""
Stream Rigid-Bodies + Labeled-Markers ▶ CSV 저장
────────────────────────────────────────────────
• 실시간 스트리밍 (Capture 없이)
• Ctrl + C  → 실시간 데이터 종료 후 → CSV 저장 (시작/종료 시간 포함)
"""

import asyncio, xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
import pandas as pd
from scipy.spatial.transform import Rotation as R
import qtm_rt as qtm

# ────────── 사용자 설정 ──────────
QTM_FILE     = None   # r"C:\Users\user\Documents\Qualisys\Data\AIM\RigidBody_Vcam_V01.qtm"   # r"C:\QTM\YourFile.qtm"  또는  None → 라이브
QTM_IP       = "127.0.0.1"
QTM_PASSWORD = "password"
WANTED_BODY  = None  # 특정 바디만: "Bat", 전부 저장하려면 None
# ──────────────────────

STAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTDIR  = Path.home() / f"Desktop/qtm_export_{STAMP}"
OUTDIR.mkdir(parents=True, exist_ok=True)
START_TIME = datetime.now()

body_rows: dict[str, list[list]] = {}
marker_rows: dict[str, list[list]] = {}

# ── 헬퍼 ──
def body_index(xml_str: str) -> dict[str, int]:
    """Get body names and their corresponding index from XML data."""
    root = ET.fromstring(xml_str)
    return {e.text.strip(): i for i, e in enumerate(root.findall("*/Body/Name"))}

def rotation_to_rpy(matrix: list[float]) -> tuple[float, float, float]:
    """3×3 회전 행렬 → Roll/Pitch/Yaw (deg). 실패 시 NaN 반환"""
    try:
        r = R.from_matrix([[matrix[0], matrix[1], matrix[2]],
                           [matrix[3], matrix[4], matrix[5]],
                           [matrix[6], matrix[7], matrix[8]]])
        return r.as_euler("xyz", degrees=True)
    except (ValueError, IndexError):
        return float("nan"), float("nan"), float("nan")

# ── 메인 ──
async def main():
    """Main function to connect to QTM and stream data."""
    conn = await qtm.connect(QTM_IP)
    if conn is None:
        print("❌ QTM 연결 실패"); return

    try:
        # QTM 연결 및 실시간 스트리밍 시작 (Capture 없이)
        async with qtm.TakeControl(conn, QTM_PASSWORD):
            # QTM 파일 재생 모드
            if QTM_FILE:
                await conn.load(QTM_FILE)
                await conn.start(rtfromfile=True)
                print(f"▶ '{QTM_FILE}' 파일 재생 중…")
            # 실시간 스트리밍 모드
            else:
                print("📡 실시간 스트리밍 시작… (Capture 없이)")
                # conn.new() 와 conn.start() 를 호출하지 않습니다.

            # Get parameters for 6D and 3D data
            bmap = body_index(await conn.get_parameters(["6d"]))
            xml3d = await conn.get_parameters(["3d"])
            m_name_list = [e.text.strip() for e in ET.fromstring(xml3d).findall("*/Marker/Name")]

            # ─ 패킷 콜백 ─
            def on_packet(pkt: qtm.QRTPacket):
                """Callback function for each received data packet."""
                fnum = pkt.framenumber  # 프레임 번호 가져오기
                _, bodies = pkt.get_6d()
                _, markers = pkt.get_3d_markers()

                if not bodies and not markers:
                    return

                # 리지드 바디 저장
                if bodies:
                    for name, idx in bmap.items():
                        if WANTED_BODY and name != WANTED_BODY: continue
                        if idx < len(bodies):
                            pos, rot = bodies[idx]
                            roll, pitch, yaw = rotation_to_rpy(rot.matrix)
                            body_rows.setdefault(name, []).append(
                                [fnum, pos.x, pos.y, pos.z, roll, pitch, yaw]
                            )

                # 마커 저장
                if markers:
                    for i, m in enumerate(markers):
                        if m is None: continue
                        key = m_name_list[i] if i < len(m_name_list) else f"M{i}"
                        x, y, z = m
                        marker_rows.setdefault(key, []).append([fnum, x, y, z])

            # Start streaming frames
            await conn.stream_frames(components=["6d", "3d"], on_packet=on_packet)

            try:
                # Keep the script running to receive data
                while True: await asyncio.sleep(1)
            except (KeyboardInterrupt, asyncio.CancelledError):
                print("\n🛑 종료 – 저장중…")

            # Stop streaming and measurement
            await conn.stream_frames_stop()
            if QTM_FILE:
                try: await conn.stop()
                except qtm.QRTCommandException: pass

    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        conn.disconnect()
        save_data()

# ── 저장 ──
def save_data():
    """Save collected data to CSV files."""
    END_TIME = datetime.now()
    if not body_rows and not marker_rows:
        print("⚠️ 수집된 데이터가 없습니다."); return

    head_time = f"# Start: {START_TIME.isoformat()}\n# End:   {END_TIME.isoformat()}\n"

    # 리지드 바디 개별 저장
    header_b = ["Frame", "Pos_X", "Pos_Y", "Pos_Z", "Roll", "Pitch", "Yaw"]
    for name, rows in body_rows.items():
        path = OUTDIR / f"{name}.csv"
        df = pd.DataFrame(rows, columns=header_b)
        df = df.drop_duplicates(subset=["Frame"]).set_index("Frame")
        if not df.empty:
            # Reindex to fill missing frames and interpolate
            full_range = range(int(df.index.min()), int(df.index.max()) + 1)
            df = df.reindex(full_range).interpolate(method='linear')
            with path.open("w", newline="") as f:
                f.write(head_time)
                df.to_csv(f, index=True)

    # 마커 통합 저장
    if marker_rows:
        all_frames = sorted({r[0] for rs in marker_rows.values() for r in rs})
        if all_frames:
            # Create a base DataFrame with all frame numbers
            base_df = pd.DataFrame(index=pd.Index(range(min(all_frames), max(all_frames) + 1), name="Frame"))
            
            for key, rows in marker_rows.items():
                df = pd.DataFrame(rows, columns=["Frame", f"{key}_Pos_X", f"{key}_Pos_Y", f"{key}_Pos_Z"])
                df = df.drop_duplicates(subset=["Frame"]).set_index("Frame")
                base_df = base_df.join(df, how='left')

            # Interpolate all columns at once
            base_df = base_df.interpolate(method='linear')
            
            path = OUTDIR / "markers.csv"
            with path.open("w", newline="") as f:
                f.write(head_time)
                base_df.to_csv(f, index=True)

    print(f"✅ 저장 완료 → {OUTDIR}")

# ── 실행 ─
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")

