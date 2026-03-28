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

st.set_page_config(page_title="Smart City Pro", page_icon="🚦", layout="wide")

st.title("🚦 Smart City Pro: Advanced Traffic Monitor")

CLASS_NAMES = {2: "Samochody", 3: "Motocykle", 5: "Autobusy", 7: "Ciężarówki"}

@st.cache_resource
def load_model():
    return YOLO('yolov8s.pt')

model = load_model()

st.sidebar.header("⚙️ Ustawienia Systemu")
device = "GPU" if torch.cuda.is_available() else "CPU"
st.sidebar.info(f"Silnik AI działa na: **{device}**")

source_url = st.sidebar.text_input(
    "Link do źródła (YouTube, .mp4, .m3u8):", 
    "https://www.youtube.com/watch?v=u7GyFcQJs98"
)

congestion_threshold = st.sidebar.slider("Próg zatoru", 2, 50, 10)

st.sidebar.markdown("---")
start_button = st.sidebar.button("▶️ Połącz i Analizuj", type="primary")
stop_button = st.sidebar.button("⏹️ Zatrzymaj")

col1, col2 = st.columns([2.5, 1.5])

with col1:
    video_placeholder = st.empty()
    st.markdown("---")
    alert_placeholder = st.empty()

with col2:
    st.markdown("### 📊 Panel Analityczny")
    stats_placeholder = st.empty() 
    st.markdown("---")
    st.markdown("### 📈 Zagęszczenie")
    traffic_chart = st.line_chart(pd.DataFrame(columns=["Zagęszczenie"]))

if start_button:
    is_youtube = "youtube.com" in source_url or "youtu.be" in source_url
    is_direct = any(ext in source_url for ext in [".mp4", ".m3u8", ".avi", ".ts"])

    if not is_youtube and not is_direct:
        st.error("Wklej poprawny link")
    else:
        with st.spinner("Łączenie..."):
            try:
                if is_direct:
                    stream_url = source_url
                else:
                    streams = streamlink.streams(source_url)
                    if not streams:
                        st.error("Błąd strumienia")
                        st.stop()
                    best_stream = streams.get('720p', streams.get('best'))
                    stream_url = best_stream.url
                    
                stream_reader = LiveStream(stream_url)
                
                counted_vehicles = {}
                last_chart_update = time.time()
                last_jam_time = 0
                last_alert_state = None 
                track_history = defaultdict(lambda: [])
                
                UI_FPS_LIMIT = 30 
                ui_refresh_interval = 1.0 / UI_FPS_LIMIT
                last_ui_update_time = time.time()
                
                st.sidebar.success("System działa.")
                
                while stream_reader.cap.isOpened() and not stop_button:
                    ret, frame = stream_reader.read()
                    if not ret or frame is None: 
                        continue 
                        
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
                    
                    if current_vehicles_in_frame >= congestion_threshold:
                        last_jam_time = current_time
                    is_jam = (current_time - last_jam_time) < 3.0
                    current_alert_state = "JAM" if is_jam else "CLEAR"
                    
                    if current_alert_state != last_alert_state:
                        if current_alert_state == "JAM":
                            alert_placeholder.markdown(f"<h3 style='text-align: center; color: #ff4b4b;'>🚨 ZATOR DROGOWY ({current_vehicles_in_frame})</h3>", unsafe_allow_html=True)
                        else:
                            alert_placeholder.markdown(f"<h3 style='text-align: center; color: #00cc66;'>🟢 Ruch płynny</h3>", unsafe_allow_html=True)
                        last_alert_state = current_alert_state 
                    
                    if current_time - last_chart_update >= 5.0:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        new_row = pd.DataFrame({"Zagęszczenie": [current_vehicles_in_frame]}, index=[timestamp])
                        traffic_chart.add_rows(new_row)
                        last_chart_update = current_time
                    
                    if current_time - last_ui_update_time >= ui_refresh_interval:
                        _, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        video_placeholder.image(buffer.tobytes(), use_container_width=True)
                        
                        with stats_placeholder.container():
                            st.metric(label="🚥 Przejechało", value=len(counted_vehicles))
                            if len(counted_vehicles) > 0:
                                df_counts = pd.Series(list(counted_vehicles.values())).value_counts()
                                st.bar_chart(df_counts, color="#FF4B4B")
                                
                        last_ui_update_time = current_time
                    
                stream_reader.release()
            except Exception as e:
                st.error(f"Błąd: {e}")
