import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import streamlit as st
import pandas as pd
from ultralytics import YOLO
import streamlink
import torch
import time
from datetime import datetime
import threading
import math
from collections import defaultdict

# ==========================================
# 0. KLASA ASYNCHRONICZNEGO CZYTNIKA
# ==========================================
class LiveStream:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.fps or self.fps != self.fps: 
            self.fps = 30.0
        self.frame_time = 1.0 / self.fps
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            start_time = time.time()
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.ret = ret
                    self.frame = cv2.resize(frame, (640, 360))
            
            elapsed = time.time() - start_time
            sleep_time = max(0, self.frame_time - elapsed)
            time.sleep(sleep_time)

    def read(self):
        return self.ret, self.frame.copy() if self.ret else None

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

# ==========================================
# 1. KONFIGURACJA STRONY
# ==========================================
st.set_page_config(page_title="Smart City Pro", page_icon="🚦", layout="wide")

st.title("🚦 Smart City Pro: Advanced Traffic Monitor")

CLASS_NAMES = {2: "Samochody 🚗", 3: "Motocykle 🏍️", 5: "Autobusy 🚌", 7: "Ciężarówki 🚛"}

@st.cache_resource
def load_model():
    return YOLO('yolov8s.pt')

model = load_model()

# ==========================================
# 2. INTERFEJS UŻYTKOWNIKA (SIDEBAR)
# ==========================================
st.sidebar.header("⚙️ Ustawienia Systemu")
device = "GPU" if torch.cuda.is_available() else "CPU"
st.sidebar.info(f"Silnik AI działa na: **{device}**")

youtube_url = st.sidebar.text_input(
    "Link do wideo (YouTube lub .mp4):", 
    "https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"
)

congestion_threshold = st.sidebar.slider("Próg zatoru (aut na ekranie)", 2, 50, 10)

st.sidebar.markdown("---")
start_button = st.sidebar.button("▶️ Połącz i Analizuj", type="primary")
stop_button = st.sidebar.button("⏹️ Zatrzymaj")

# ==========================================
# 3. GŁÓWNY WIDOK (KOLUMNY)
# ==========================================
col1, col2 = st.columns([2.5, 1.5])

with col1:
    video_placeholder = st.empty()
    st.markdown("---")
    alert_placeholder = st.empty()

with col2:
    st.markdown("### 📊 Panel Analityczny na żywo")
    stats_placeholder = st.empty() 
    st.markdown("---")
    st.markdown("### 📈 Zagęszczenie ruchu")
    traffic_chart = st.line_chart(pd.DataFrame(columns=["Zagęszczenie"]))

# ==========================================
# 4. GŁÓWNA PĘTLA ANALITYCZNA
# ==========================================
if start_button:
    # Zmieniony warunek, aby przepuszczał linki .mp4
    if "youtube.com" not in youtube_url and "youtu.be" not in youtube_url and not any(ext in youtube_url for ext in [".mp4", ".m3u8", ".avi"]):
        st.error("Wklej poprawny link do YouTube lub bezpośredni strumień kamery (.mp4, .m3u8)")
    else:
        with st.spinner("Łączenie ze strumieniem..."):
            try:
                # Omijamy streamlink dla bezpośrednich strumieni wideo
                if any(ext in youtube_url for ext in [".mp4", ".m3u8", ".avi"]):
                    stream_url = youtube_url
                else:
                    streams = streamlink.streams(youtube_url)
                    if not streams:
                        st.error("Nie można pobrać wideo. Upewnij się, że to transmisja na żywo")
                        st.stop()
                    best_stream = streams.get('720p', streams.get('best'))
                    stream_url = best_stream.url
                    
                stream_reader = LiveStream(stream_url)
                
                counted_vehicles = {}
                history_data = []
                last_save_time = time.time()
                last_jam_time = 0
                last_alert_state = None 
                track_history = defaultdict(lambda: [])
                
                # --- LIMITER ODŚWIEŻANIA UI ---
                UI_FPS_LIMIT = 15 # Renderujemy stronę maksymalnie 15 razy na sekundę
                ui_refresh_interval = 1.0 / UI_FPS_LIMIT
                last_ui_update_time = time.time()
                
                st.sidebar.success("System połączony. Zoptymalizowano UI (Throttling aktywny).")
                time.sleep(1)
                
                while stream_reader.cap.isOpened() and not stop_button:
                    ret, frame = stream_reader.read()
                    
                    # Zabezpieczenie przed "cichym zawieszeniem"
                    if not ret or frame is None: 
                        st.error("❌ Nie można odczytać klatki wideo. Strumień się zakończył, YouTube zablokował połączenie lub wystąpił błąd sieci.")
                        break 
                        
                    # Silnik AI pracuje na każdej klatce z pełną prędkością
                    results = model.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False)
                    annotated_frame = results[0].plot()
                    
                    current_vehicles_in_frame = 0
                    
                    if results[0].boxes.id is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                        classes = results[0].boxes.cls.cpu().numpy().astype(int)
                        
                        for box, track_id, cls_id in zip(boxes, track_ids, classes):
                            x1, y1, x2, y2 = box
                            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                            
                            track = track_history[track_id]
                            track.append((cx, cy))
                            
                            if len(track) > 30:
                                track.pop(0)
                                
                            start_x, start_y = track[0]
                            total_distance = math.hypot(cx - start_x, cy - start_y)
                            
                            if total_distance > 30:
                                current_vehicles_in_frame += 1
                                
                                for i in range(1, len(track)):
                                    thickness = int(math.sqrt(float(i)) * 1.5)
                                    cv2.line(annotated_frame, track[i - 1], track[i], (0, 215, 255), thickness)

                                if total_distance > 60 and track_id not in counted_vehicles:
                                    counted_vehicles[track_id] = CLASS_NAMES.get(cls_id, "Inne")
                    
                    current_time = time.time()
                    
                    # --- LOGIKA ALERTÓW ---
                    if current_vehicles_in_frame >= congestion_threshold:
                        last_jam_time = current_time
                        
                    is_jam = (current_time - last_jam_time) < 3.0
                    current_alert_state = "JAM" if is_jam else "CLEAR"
                    
                    if current_alert_state != last_alert_state:
                        if current_alert_state == "JAM":
                            alert_placeholder.markdown(f"<h3 style='text-align: center; color: #ff4b4b;'>🚨 ZATOR DROGOWY (Aut w ruchu: {current_vehicles_in_frame})</h3>", unsafe_allow_html=True)
                        else:
                            alert_placeholder.markdown(f"<h3 style='text-align: center; color: #00cc66;'>🟢 Ruch odbywa się płynnie</h3>", unsafe_allow_html=True)
                        last_alert_state = current_alert_state 
                    
                    # --- ZAPIS W TLE ---
                    if current_time - last_save_time >= 5.0:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        history_data.append({"Czas": timestamp, "Zagęszczenie": current_vehicles_in_frame})
                        
                        data_copy = list(history_data)
                        threading.Thread(target=lambda: pd.DataFrame(data_copy).to_csv("traffic_history.csv", index=False)).start()
                        
                        new_row = pd.DataFrame({"Zagęszczenie": [current_vehicles_in_frame]}, index=[timestamp])
                        traffic_chart.add_rows(new_row)
                        
                        last_save_time = current_time
                    
                    # ==========================================
                    # OPTYMALIZACJA WYŚWIETLANIA (UI THROTTLING)
                    # ==========================================
                    if current_time - last_ui_update_time >= ui_refresh_interval:
                        _, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                        video_placeholder.image(buffer.tobytes(), use_container_width=True)
                        
                        with stats_placeholder.container():
                            st.metric(label="🚥 Razem przejechało", value=len(counted_vehicles))
                            if len(counted_vehicles) > 0:
                                df_counts = pd.Series(list(counted_vehicles.values())).value_counts()
                                st.bar_chart(df_counts, color="#FF4B4B")
                                
                        last_ui_update_time = current_time
                    
                stream_reader.release()
            except Exception as e:
                st.error(f"Wystąpił błąd: {e}")
