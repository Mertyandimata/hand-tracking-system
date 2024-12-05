import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from enum import Enum
import time
from collections import deque
from typing import List, Tuple, Dict
from scipy.signal import savgol_filter

class HandState(Enum):
    """Basit el durumları"""
    IDLE = "IDLE"     # El açık/doğal pozisyon
    GRAB = "GRAB"     # Kavrama hareketi

@dataclass
class HandInfo:
    """Gelişmiş el bilgileri"""
    state: HandState          # El durumu
    hand_type: str           # Sağ/Sol el
    confidence: float        # Güven skoru
    position: np.ndarray     # Filtrelenmiş el pozisyonu
    velocity: float          # El hızı
    stability: float         # Hareket stabilitesi
    fingers_closed: int      # Kapalı parmak sayısı

class HandTracking:
    def __init__(self):
        # MediaPipe ayarları
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        
        # Kalman filtresi kurulumu
        self.kf_hands = {}  # Her el için ayrı Kalman filtresi
        
        # Hareket geçmişi
        self.position_history = {}  # Her el için pozisyon geçmişi
        self.state_history = {}    # Her el için durum geçmişi
        
        # UI ayarları
        self.colors = {
            'bg': (10, 10, 25),
            'panel': (20, 20, 35),
            'text': (240, 240, 240),
            'accent': (0, 255, 200),
            'grab': (255, 100, 100),
            'idle': (100, 255, 100)
        }
        
        self.fps_history = deque(maxlen=30)
        self.prev_frame_time = 0

    def setup_kalman(self, hand_id: str):
        """Her el için Kalman filtresi kur"""
        kf = cv2.KalmanFilter(6, 3)  # 6 state vars (x,y,z ve hızları), 3 measurement
        kf.measurementMatrix = np.array([[1,0,0,0,0,0],
                                       [0,1,0,0,0,0],
                                       [0,0,1,0,0,0]], np.float32)
        
        kf.transitionMatrix = np.array([[1,0,0,1,0,0],
                                      [0,1,0,0,1,0],
                                      [0,0,1,0,0,1],
                                      [0,0,0,1,0,0],
                                      [0,0,0,0,1,0],
                                      [0,0,0,0,0,1]], np.float32)
        
        kf.processNoiseCov = np.array([[1,0,0,0,0,0],
                                     [0,1,0,0,0,0],
                                     [0,0,1,0,0,0],
                                     [0,0,0,1,0,0],
                                     [0,0,0,0,1,0],
                                     [0,0,0,0,0,1]], np.float32) * 0.03
        
        self.kf_hands[hand_id] = kf
        self.position_history[hand_id] = deque(maxlen=30)
        self.state_history[hand_id] = deque(maxlen=10)

    def filter_position(self, hand_id: str, position: np.ndarray) -> np.ndarray:
        """Pozisyon filtreleme (Kalman + Savitzky-Golay)"""
        if hand_id not in self.kf_hands:
            self.setup_kalman(hand_id)
        
        kf = self.kf_hands[hand_id]
        measurement = position.reshape(3, 1).astype(np.float32)
        
        # Kalman filtresi uygula
        kf.correct(measurement)
        prediction = kf.predict()
        filtered_position = prediction[:3].reshape(-1)
        
        # Savitzky-Golay filtresi uygula
        self.position_history[hand_id].append(filtered_position)
        if len(self.position_history[hand_id]) >= 5:
            positions = np.array(list(self.position_history[hand_id]))
            filtered_position = savgol_filter(
                np.vstack([positions, filtered_position]), 
                window_length=5, 
                polyorder=2, 
                axis=0
            )[-1]
        
        return filtered_position

    def calculate_velocity(self, hand_id: str, current_position: np.ndarray) -> float:
        """El hızını hesapla"""
        if not self.position_history[hand_id]:
            return 0.0
            
        prev_position = self.position_history[hand_id][-1]
        velocity = np.linalg.norm(current_position - prev_position)
        return velocity

    def calculate_stability(self, hand_id: str, current_state: HandState) -> float:
        """Hareket stabilitesi hesapla"""
        if not self.state_history[hand_id]:
            return 1.0
            
        recent_states = list(self.state_history[hand_id])
        same_state_count = sum(1 for state in recent_states if state == current_state)
        return same_state_count / len(recent_states)

    def detect_hand_state(self, landmarks, velocity: float) -> Tuple[HandState, float]:
        """
        Geliştirilmiş el durumu tespiti
        Parmak pozisyonları ve hız bilgisini kullanır
        """
        finger_tips = [8, 12, 16, 20]
        finger_bases = [5, 9, 13, 17]
        thumb_tip = 4
        thumb_base = 2
        
        closed_fingers = 0
        
        # Başparmak kontrolü
        if landmarks[thumb_tip].y > landmarks[thumb_base].y:
            closed_fingers += 1
        
        # Diğer parmaklar için kontrol
        for tip, base in zip(finger_tips, finger_bases):
            if landmarks[tip].y > landmarks[base].y:
                closed_fingers += 1
        
        # Durum tespiti (hız bilgisini de kullan)
        if closed_fingers >= 4:
            confidence = 0.9 if velocity < 0.5 else 0.8
            return HandState.GRAB, confidence
        else:
            confidence = 0.9 if velocity < 0.5 else 0.8
            return HandState.IDLE, confidence

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[HandInfo]]:
        """Frame'i işle ve el bilgilerini döndür"""
        height, width = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        # Arka plan efekti
        overlay = np.zeros_like(frame)
        overlay[:] = self.colors['bg']
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
        
        hand_info_list = []
        
        if results.multi_hand_landmarks:
            for idx, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)):
                
                # El tipi ve ID
                hand_type = "Right" if handedness.classification[0].label == "Right" else "Left"
                hand_id = f"{hand_type}_{idx}"
                
                # Landmark pozisyonlarını numpy dizisine dönüştür
                landmarks = np.array([[lm.x * width, lm.y * height, lm.z] 
                                    for lm in hand_landmarks.landmark])
                
                # El pozisyonunu filtrele
                palm_center = np.mean(landmarks[[0, 5, 9, 13, 17]], axis=0)
                filtered_position = self.filter_position(hand_id, palm_center)
                
                # Hız ve stabilite hesapla
                velocity = self.calculate_velocity(hand_id, filtered_position)
                
                # El durumunu tespit et
                state, confidence = self.detect_hand_state(hand_landmarks.landmark, velocity)
                stability = self.calculate_stability(hand_id, state)
                
                # Parmak durumunu hesapla
                closed_fingers = sum(1 for tip, base in zip([8,12,16,20], [5,9,13,17])
                                  if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[base].y)
                
                # El bilgilerini kaydet
                hand_info = HandInfo(
                    state=state,
                    hand_type=hand_type,
                    confidence=confidence,
                    position=filtered_position,
                    velocity=velocity,
                    stability=stability,
                    fingers_closed=closed_fingers
                )
                hand_info_list.append(hand_info)
                
                # Geçmişi güncelle
                self.state_history[hand_id].append(state)
                
                # El iskeletini çiz
                self.draw_hand_landmarks(frame, hand_landmarks, hand_info)
        
        # UI'ı güncelle
        frame = self.draw_ui(frame, hand_info_list)
        return frame, hand_info_list

    def draw_hand_landmarks(self, frame: np.ndarray, landmarks, hand_info: HandInfo):
        """Gelişmiş el iskeleti çizimi"""
        color = self.colors['grab'] if hand_info.state == HandState.GRAB else self.colors['idle']
        
        # Ana iskelet
        self.mp_draw.draw_landmarks(
            frame, landmarks, self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=color, thickness=3),
            self.mp_draw.DrawingSpec(color=color, thickness=3)
        )
        
        # Stabilite göstergesi (daire)
        center = tuple(map(int, [
            landmarks.landmark[0].x * frame.shape[1],
            landmarks.landmark[0].y * frame.shape[0]
        ]))
        radius = int(30 * hand_info.stability)
        cv2.circle(frame, center, radius, color, 2)

    def draw_ui(self, frame: np.ndarray, hand_info_list: List[HandInfo]) -> np.ndarray:
        """Modern ve bilgilendirici UI"""
        height, width = frame.shape[:2]
        panel_height = 140
        
        # Üst panel
        panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        panel[:] = self.colors['panel']
        
        # FPS hesapla
        current_time = time.time()
        fps = 1 / (current_time - self.prev_frame_time)
        self.prev_frame_time = current_time
        self.fps_history.append(fps)
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        
        # FPS göster
        cv2.putText(panel, f"FPS: {avg_fps:.1f}", 
                   (width - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['accent'], 2)
        
        # El bilgilerini göster
        for idx, hand_info in enumerate(hand_info_list):
            y_pos = 35 + (idx * 45)
            color = self.colors['grab'] if hand_info.state == HandState.GRAB else self.colors['idle']
            
            # Durum metni
            status_text = f"{hand_info.hand_type}: {hand_info.state.value}"
            cv2.putText(panel, status_text, (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Detaylı bilgiler
            details = f"Conf: {hand_info.confidence:.2f} | Stab: {hand_info.stability:.2f}"
            cv2.putText(panel, details, (300, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 1)
            
            # Hız göstergesi
            vel_text = f"Vel: {hand_info.velocity:.2f}"
            cv2.putText(panel, vel_text, (600, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['accent'], 1)
        
        # Paneli frame'e ekle
        frame[0:panel_height, 0:width] = cv2.addWeighted(
            frame[0:panel_height, 0:width], 0.3,
            panel, 0.7,
            0
        )
        
        return frame

    def run(self):
        """Ana çalışma döngüsü"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Gelişmiş El Takip Sistemi başlatılıyor...")
        print("Çıkmak için 'q' tuşuna basın")
        
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    continue
                
                processed_frame, _ = self.process_frame(frame)
                cv2.imshow('Advanced Hand Tracking', processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"Hata oluştu: {str(e)}")
            
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()

if __name__ == "__main__":
    tracker = HandTracking()
    tracker.run()