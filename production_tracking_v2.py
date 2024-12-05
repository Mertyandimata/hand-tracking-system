import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import time
from collections import defaultdict, deque  # Added deque import
import threading
from datetime import datetime, timedelta
import json
import logging
import os
import contextlib
from hand_tracking import HandTracking, HandState, HandInfo
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='production_tracking.log'
)

class ProductionState(Enum):
    """Production states for workflow tracking"""
    IDLE = "Idle"
    OBJECT_DETECTED = "Object Detected"
    PROCESSING_STARTED = "Processing Started"
    PROCESSING = "Processing"
    QUALITY_CHECK = "Quality Check"
    COMPLETED = "Completed"

@dataclass
class ProductionStep:
    """Represents a dynamically created production step"""
    step_number: int
    start_time: float
    total_active_time: float = 0.0
    last_active_time: float = 0.0
    end_time: Optional[float] = None
    status: str = "Active"
    
    def calculate_progress(self) -> float:
        """Calculate step progress - now based on active working time"""
        if self.end_time:
            return 100.0
        return min(100.0, (self.total_active_time / 30.0) * 100)  # Using 30s as reference

@dataclass
class ProductionCycle:
    """Tracks a production cycle with dynamic steps"""
    id: str
    object_class: str
    start_time: float
    steps: List[ProductionStep]
    current_step_index: int = 0
    state: ProductionState = ProductionState.IDLE
    hands_used: Set[str] = field(default_factory=set)
    total_processing_time: float = 0.0
    quality_score: float = 0.0
    last_grab_time: float = 0.0
    grab_timeout: float = 10.0  # Timeout window in seconds
    
    @property
    def current_step(self) -> Optional[ProductionStep]:
        """Get current production step"""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None
    
    def should_create_new_step(self, current_time: float) -> bool:
        """Determine if a new step should be created based on timing"""
        if not self.steps:
            return True
        return (current_time - self.last_grab_time) > self.grab_timeout
    
    def add_new_step(self, current_time: float) -> None:
        """Add a new step to the workflow"""
        if self.current_step:
            self.current_step.end_time = current_time
            self.current_step.status = "Completed"
        
        new_step = ProductionStep(
            step_number=len(self.steps) + 1,
            start_time=current_time,
            last_active_time=current_time
        )
        self.steps.append(new_step)
        self.current_step_index = len(self.steps) - 1
    
    def to_dict(self) -> Dict:
        """Convert cycle data to dictionary for logging"""
        return {
            'id': self.id,
            'object_class': self.object_class,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'total_time': self.total_processing_time,
            'state': self.state.value,
            'hands_used': list(self.hands_used),
            'quality_score': self.quality_score,
            'steps': [
                {
                    'name': step.name,
                    'expected_duration': step.expected_duration,
                    'actual_duration': step.actual_duration,
                    'status': step.status
                }
                for step in self.steps
            ]
        }

class ProductionAnalytics:
    """Handles production statistics and analysis"""
    def __init__(self):
        self.cycles: List[ProductionCycle] = []
        self.object_stats: Dict[str, Dict] = defaultdict(
            lambda: {'count': 0, 'avg_time': 0.0, 'quality_scores': []}
        )
    
    def add_cycle(self, cycle: ProductionCycle):
        """Add completed cycle and update statistics"""
        self.cycles.append(cycle)
        stats = self.object_stats[cycle.object_class]
        stats['count'] += 1
        stats['quality_scores'].append(cycle.quality_score)
        
        # Update average processing time
        prev_avg = stats['avg_time']
        stats['avg_time'] = (prev_avg * (stats['count'] - 1) + 
                           cycle.total_processing_time) / stats['count']
        
        # Log cycle completion
        logging.info(f"Cycle completed: {json.dumps(cycle.to_dict(), indent=2)}")

class ProductionTracking:
    """Main production tracking system"""
    def __init__(self, hand_tracking):
            """
            Initialize the production tracking system with optimized configurations.
            
            Args:
                hand_tracking: Initialized HandTracking instance for gesture recognition
            """
            self.hand_tracking = hand_tracking
            self.is_step_active = False
            self.current_step = 1
            self.step_start_time = 0
            self.active_time = 0
            self.last_grab_time = 0
            self.hand_stabilization_threshold = 4.0  # 4 saniye stabilizasyon süresi
            self.hand_first_detected_time = None
            self.is_hand_stable = False
            # Initialize YOLO11 model with error handling
            print("Initializing YOLO11 model...")
            try:
                self.model = YOLO('yolo11n.pt')
                print("✓ YOLO11 model loaded successfully")
            except Exception as e:
                print(f"✗ Error loading YOLO11 model: {e}")
                raise

            # Detection and interaction configurations
            self.detection_config = {
                'hand_radius': 300,        # Pixel radius for hand-object interaction
                'person_radius': 350,      # Person detection radius from center
                'interaction_conf': 0.25,   # Minimum detection confidence
                'tracking_persist': 5,      # Frames to maintain tracking
                'max_objects': 3,          # Maximum simultaneous objects to track
                'timeout_duration': 2.0,    # Seconds before interaction timeout
            }

            # Production workflow configuration
            self.default_workflow = [
                ProductionStep("Object Detection & Pickup", 20.0),
                ProductionStep("Initial Processing", 35.0),
                ProductionStep("Assembly/Manipulation", 50.0),
                ProductionStep("Quality Verification", 25.0)
            ]

            # State management and analytics
            self.analytics = ProductionAnalytics()
            self.current_cycle: Optional[ProductionCycle] = None
            self.last_interaction_time = time.time()
            self.state_lock = threading.Lock()
            
            # Performance monitoring
            self.frame_times = deque(maxlen=30)
            self.start_time = time.time()
            self.frames_processed = 0
            self.GRAB_TIMEOUT = 20.0
            # Object tracking history
            self.object_history = deque(maxlen=self.detection_config['tracking_persist'])
            self.current_objects = set()
            self.cycle_timeout = 30.0
            # UI configuration with enhanced color scheme
            self.colors = {
                'bg': (10, 10, 25),
                'panel': (20, 20, 35),
                'text': (240, 240, 240),
                'accent': (0, 255, 200),
                'warning': (255, 165, 0),
                'success': (100, 255, 100),
                'error': (255, 70, 70),
                'active': (0, 255, 0),
                'inactive': (200, 200, 200),
                'interaction': (0, 200, 255),
                'highlight': (255, 200, 0)
            }

    def analyze_interactions(self, frame: np.ndarray, hand_info: List[HandInfo]) -> List[Dict]:
        """Analyze and detect objects near hands"""
        results = self.model(
            frame, 
            stream=True,
            conf=self.detection_config['interaction_conf']
        )
        
        # Get hand positions for interaction analysis
        hand_positions = [
            hand.position[:2] for hand in hand_info
            if hand.state == HandState.GRAB
        ]
        
        detected_objects = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                # Skip person class
                if result.names[int(box.cls[0])] == 'person':
                    continue
                    
                detection = {
                    'xyxy': box.xyxy[0],
                    'conf': float(box.conf[0]),
                    'class': result.names[int(box.cls[0])],
                    'center': np.mean(box.xyxy[0].reshape((2, 2)), axis=0),
                }
                
                # Calculate interaction with hands
                if hand_positions:
                    for hand_pos in hand_positions:
                        dist_to_hand = np.linalg.norm(detection['center'] - hand_pos)
                        if dist_to_hand < self.detection_config['hand_radius']:
                            detection['interaction_score'] = 1.0 - (dist_to_hand / self.detection_config['hand_radius'])
                            detected_objects.append(detection)
                            break
        
        return detected_objects

    def _draw_detection_box(self, frame: np.ndarray, detection: Dict) -> np.ndarray:
        """Draw elegant detection box with dashed lines"""
        x1, y1, x2, y2 = map(int, detection['xyxy'])
        
        # Calculate colors based on confidence
        color = (0, 255, 200)  # Turkuaz tonu
        thickness = 2
        
        # Draw dashed rectangle
        def draw_dashed_line(img, pt1, pt2, color, thickness, gap=10):
            dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
            pts = []
            for i in np.arange(0, dist, gap):
                r = i / dist
                x = int((pt1[0] * (1 - r) + pt2[0] * r))
                y = int((pt1[1] * (1 - r) + pt2[1] * r))
                pts.append((x, y))

            for i in range(len(pts) - 1):
                if i % 2 == 0:
                    cv2.line(img, pts[i], pts[i + 1], color, thickness)
        
        # Draw dashed rectangle
        draw_dashed_line(frame, (x1, y1), (x2, y1), color, thickness)  # Top
        draw_dashed_line(frame, (x2, y1), (x2, y2), color, thickness)  # Right
        draw_dashed_line(frame, (x1, y2), (x2, y2), color, thickness)  # Bottom
        draw_dashed_line(frame, (x1, y1), (x1, y2), color, thickness)  # Left
        
        # Add elegant label
        label = f"{detection['class']} {detection['conf']:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        
        # Draw semi-transparent background for label
        cv2.rectangle(frame, 
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0] + 10, y1),
                    color, -1)
        
        # Add text
        cv2.putText(frame, label,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1)
        
        return frame
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        # Get hand tracking information
        hand_info = self.hand_tracking.process_frame(frame)
        
        # Create a region of interest (ROI) around the hands
        roi = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        for hand in hand_info:
            x, y, w, h = hand.bbox
            roi[y:y+h, x:x+w] = 255
        
        # Apply object detection within the ROI
        results = self.model(frame, roi=roi)
        detections = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                detection = {
                    'xyxy': box.xyxy[0],
                    'conf': box.conf[0],
                    'class': result.names[int(box.cls[0])]
                }
                detections.append(detection)
        
        return detections

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[HandInfo]]:
        try:
            # Hand tracking ve kopya frame
            hand_processed_frame, hand_info = self.hand_tracking.process_frame(frame.copy())
            
            # Nesne tespiti
            #detected_objects = self.analyze_interactions(frame, hand_info)
            
            # Her tespit için çerçeve çizimi 
            #for detection in detected_objects:
            #    hand_processed_frame = self._draw_detection_box(hand_processed_frame, detection)
            
            current_time = time.time()
            
            # El yakalama kontrolü
            hand_grabbing = any(info.state == HandState.GRAB for info in hand_info)
            
            if not hasattr(self, 'last_grab_time'):
                self.last_grab_time = current_time

            # El stabilizasyonu
            if hand_grabbing:
                if not hasattr(self, 'hand_first_detected_time'):
                    self.hand_first_detected_time = current_time
                    self.is_hand_stable = False
                
                stabilization_time = current_time - self.hand_first_detected_time
                
                if stabilization_time >= self.hand_stabilization_threshold:
                    self.is_hand_stable = True
            else:
                if hasattr(self, 'hand_first_detected_time'):
                    del self.hand_first_detected_time
                self.is_hand_stable = False

            # Adım mantığı
            if hasattr(self, 'is_hand_stable') and self.is_hand_stable:
                time_since_last_grab = current_time - self.last_grab_time
                
                if hand_grabbing:
                    if not self.is_step_active:
                        self.step_start_time = current_time
                        self.is_step_active = True
                        self.active_time = 0
                    
                    self.active_time = current_time - self.step_start_time
                    
                    if time_since_last_grab <= self.GRAB_TIMEOUT:
                        self.last_grab_time = current_time
                    else:
                        self.current_step += 1
                        self.step_start_time = current_time
                        self.active_time = 0
                        self.last_grab_time = current_time
                else:
                    if self.is_step_active and time_since_last_grab > self.GRAB_TIMEOUT:
                        self.is_step_active = False
                        self.active_time = 0
            
            # UI çizimi
            final_frame = self._draw_ui(hand_processed_frame, current_time)
            
            return final_frame, hand_info
            
        except Exception as e:
            logging.error(f"Frame processing error: {str(e)}")
            return frame, []
    def _draw_ui(self, frame: np.ndarray, current_time: float) -> np.ndarray:
        """
        Draw a more refined UI showing step number and active time with stabilization indicator
        """
        # Create semi-transparent overlay for text background
        height, width = frame.shape[:2]
        overlay = frame.copy()
        
        # Draw step information box
        box_height = 240  # Increased height to accommodate stabilization
        box_width = 300
        box_x = width - box_width - 30
        box_y = 30
        
        cv2.rectangle(overlay, 
                    (box_x, box_y), 
                    (box_x + box_width, box_y + box_height),
                    (30, 30, 30),  # Darker background for better contrast
                    -1)
        
        # Stabilization Logic Display
        if hasattr(self, 'hand_first_detected_time') and not getattr(self, 'is_hand_stable', False):
            # Calculate remaining stabilization time
            stabilization_time = current_time - self.hand_first_detected_time
            remaining_time = max(0, self.hand_stabilization_threshold - stabilization_time)
            
            # Stabilization Warning
            cv2.putText(overlay, "STABILIZING HANDS",
                        (box_x + 10, box_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        self.colors['warning'],
                        2)
            
            # Remaining Time
            cv2.putText(overlay, f"Waiting: {remaining_time:.1f}s",
                        (box_x + 10, box_y + 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        self.colors['text'],
                        2)
        
        # Calculate time since last grab
        time_since_last_grab = current_time - self.last_grab_time
        is_active = time_since_last_grab <= self.GRAB_TIMEOUT and getattr(self, 'is_hand_stable', False)
        
        # Draw step number
        step_text = f"Cycle {self.current_step}"
        cv2.putText(overlay, step_text,
                (box_x + 10, box_y + 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.colors['active'] if is_active else self.colors['inactive'],
                2)
        
        # Draw active time
        time_text = f"Active Time: {self.active_time:.1f}s"
        cv2.putText(overlay, time_text,
                (box_x + 10, box_y + 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                self.colors['text'],
                2)
        
        # Draw remaining time
        remaining_text = f"Next Step: {max(0, self.GRAB_TIMEOUT - time_since_last_grab):.1f}s"
        cv2.putText(overlay, remaining_text,
                (box_x + 10, box_y + 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.colors['warning'],
                2)
        
        # Status indicator
        status = "ACTIVE" if is_active else "WAITING"
        cv2.putText(overlay, status,
                (box_x + 10, box_y + 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.colors['active'] if is_active else self.colors['inactive'],
                2)
        
        # Blend overlay with original frame
        final_frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
        
        return final_frame

    def _update_performance_metrics(self, frame_start: float):
        """
        Update system performance metrics.
        
        Args:
            frame_start: Timestamp when frame processing started
        """
        self.frames_processed += 1
        process_time = time.time() - frame_start
        self.frame_times.append(process_time)
        
        # Calculate and log performance metrics every 100 frames
        if self.frames_processed % 100 == 0:
            avg_fps = len(self.frame_times) / sum(self.frame_times)
            logging.info(f"Performance metrics - FPS: {avg_fps:.1f}")
    def _create_visualization(
        self, 
        frame: np.ndarray, 
        hand_objects: List[Dict],
        relevant_person: Optional[Dict],
        hand_info: List[HandInfo]
    ) -> np.ndarray:
        """
        Create comprehensive visualization of all tracked elements.
        
        Args:
            frame: Base frame to draw on
            hand_objects: Detected objects near hands
            relevant_person: Information about relevant person
            hand_info: Hand tracking information
            
        Returns:
            Frame with all visualizations applied
        """
        # Draw object detections
        for obj in hand_objects:
            x1, y1, x2, y2 = map(int, obj['xyxy'])
            
            # Calculate visualization color based on interaction score
            interaction_color = tuple(map(
                lambda x: int(x * obj.get('interaction_score', 0.5)),
                self.colors['interaction']
            ))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), interaction_color, 2)
            
            # Draw label with enhanced information
            label = f"{obj['class']} ({obj['conf']:.2f})"
            label_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            cv2.rectangle(
                frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                interaction_color, -1
            )
            
            cv2.putText(
                frame, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2
            )

        # Draw relevant person if detected
        if relevant_person:
            x1, y1, x2, y2 = map(int, relevant_person['xyxy'])
            cv2.rectangle(
                frame,
                (x1, y1), (x2, y2),
                self.colors['accent'], 2
            )
        
        # Add workflow UI
        frame_with_ui = self.draw_workflow_ui(frame)
        
        return frame_with_ui

    
    def update_workflow_state(self, hand_info_list: List, detections: List[Dict]):
        """
        Update workflow state with dynamic step creation and management
        """
        with self.state_lock:
            hands_grabbing = [info for info in hand_info_list 
                            if info.state == HandState.GRAB]
            
            current_time = time.time()
            
            # Handle state transitions
            if not self.current_cycle:
                if hands_grabbing and detections:
                    # Start new cycle
                    self.current_cycle = ProductionCycle(
                        id=f"PROD_{int(current_time)}",
                        object_class=detections[0]['class'],
                        start_time=current_time,
                        steps=[],
                        last_grab_time=current_time
                    )
                    self.current_cycle.state = ProductionState.OBJECT_DETECTED
                    self.current_cycle.add_new_step(current_time)
                    logging.info(f"New cycle started: {self.current_cycle.object_class}")
            
            elif self.current_cycle:
                # Update hands used
                for hand in hands_grabbing:
                    self.current_cycle.hands_used.add(hand.hand_type)
                
                if hands_grabbing:
                    # Check if we need to create a new step
                    if self.current_cycle.should_create_new_step(current_time):
                        self.current_cycle.add_new_step(current_time)
                    
                    # Update current step timing
                    current_step = self.current_cycle.current_step
                    if current_step:
                        time_diff = current_time - current_step.last_active_time
                        current_step.total_active_time += time_diff
                        current_step.last_active_time = current_time
                        self.current_cycle.last_grab_time = current_time
                
                # Check for cycle completion or timeout
                if not hands_grabbing and not detections:
                    inactive_time = current_time - self.current_cycle.last_grab_time
                    if inactive_time > self.cycle_timeout:
                        # Complete the cycle
                        if self.current_cycle.current_step:
                            self.current_cycle.current_step.end_time = current_time
                            self.current_cycle.current_step.status = "Completed"
                        
                        self.current_cycle.state = ProductionState.COMPLETED
                        self.current_cycle.total_processing_time = (
                            current_time - self.current_cycle.start_time
                        )
                        self.analytics.add_cycle(self.current_cycle)
                        self.current_cycle = None
                        logging.info("Production cycle completed due to timeout")

    def calculate_quality_score(self, cycle: ProductionCycle) -> float:
        """
        Calculate quality score based on timing and completion
        """
        score = 100.0
        
        for step in cycle.steps:
            if step.actual_duration > 0:
                # Penalize for time overrun
                time_ratio = step.actual_duration / step.expected_duration
                if time_ratio > 1.0:
                    score -= min(20, (time_ratio - 1.0) * 20)
        
        return max(0.0, score)

    def draw_workflow_ui(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw comprehensive production tracking UI with detailed metrics
        
        Args:
            frame (np.ndarray): Input frame to add UI elements to
            
        Returns:
            np.ndarray: Frame with added UI panel
        """
        height, width = frame.shape[:2]
        panel_width = 350
        
        # Create side panel
        panel = np.zeros((height, panel_width, 3), dtype=np.uint8)
        panel[:] = self.colors['panel']
        
        y_pos = 30
        
        # Draw current cycle information
        if self.current_cycle:
            # Header section
            y_pos = self._draw_cycle_header(panel, y_pos)
            
            # Current state and progress
            y_pos = self._draw_cycle_progress(panel, y_pos)
            
            # Step-by-step workflow
            y_pos = self._draw_workflow_steps(panel, y_pos)
            
            # Performance metrics
            y_pos = self._draw_performance_metrics(panel, y_pos)
        else:
            cv2.putText(panel, "Waiting for production...",
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, self.colors['text'], 1)
        
        # Analytics section at bottom
        y_pos = height - 200
        y_pos = self._draw_analytics_section(panel, y_pos)
        
        # Combine panel with frame
        combined = np.zeros((height, width + panel_width, 3), dtype=np.uint8)
        combined[:, :width] = frame
        combined[:, width:] = panel
        
        return combined

    def _draw_cycle_header(self, panel: np.ndarray, y_pos: int) -> int:
        """Draw cycle identification and basic info section"""
        # Cycle ID
        cv2.putText(panel, f"Production Cycle: {self.current_cycle.id}",
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   self.colors['accent'], 2)
        y_pos += 30
        
        # Object type
        cv2.putText(panel, f"Object: {self.current_cycle.object_class}",
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   self.colors['text'], 1)
        y_pos += 25
        
        # Hands used
        hands_text = f"Hands: {', '.join(self.current_cycle.hands_used)}"
        cv2.putText(panel, hands_text, (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   self.colors['text'], 1)
        y_pos += 35
        
        return y_pos

    def _draw_cycle_progress(self, panel: np.ndarray, y_pos: int) -> int:
        """Draw current cycle state and overall progress"""
        # Current state
        state_color = (self.colors['success'] 
                      if self.current_cycle.state == ProductionState.COMPLETED
                      else self.colors['warning'])
        cv2.putText(panel, f"State: {self.current_cycle.state.value}",
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   state_color, 2)
        y_pos += 35
        
        # Overall progress bar
        total_expected = sum(step.expected_duration for step in self.current_cycle.steps)
        total_actual = sum(step.actual_duration for step in self.current_cycle.steps)
        progress = min(100, (total_actual / total_expected) * 100)
        
        cv2.putText(panel, f"Overall Progress: {progress:.1f}%",
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   self.colors['text'], 1)
        y_pos += 25
        
        # Draw overall progress bar
        bar_width = 330
        bar_height = 15
        cv2.rectangle(panel,
                     (10, y_pos),
                     (10 + bar_width, y_pos + bar_height),
                     self.colors['text'], 1)
        
        if progress > 0:
            filled_width = int(bar_width * progress/100)
            cv2.rectangle(panel,
                         (10, y_pos),
                         (10 + filled_width, y_pos + bar_height),
                         self.colors['accent'], -1)
        y_pos += 30
        
        return y_pos

    def _draw_workflow_steps(self, panel: np.ndarray, y_pos: int) -> int:
        """Draw dynamic workflow steps progress"""
        if not self.current_cycle or not self.current_cycle.steps:
            return y_pos
        
        current_time = time.time()
        
        # Calculate panel layout based on number of steps
        steps = self.current_cycle.steps
        available_height = panel.shape[0] - y_pos - 20
        step_height = min(80, available_height // len(steps))
        
        for step in steps:
            is_current = step == self.current_cycle.current_step
            is_completed = step.end_time is not None
            
            # Determine step color based on state
            if is_completed:
                color = self.colors['success']
            elif is_current:
                time_since_last_grab = current_time - self.current_cycle.last_grab_time
                color = (self.colors['accent'] if time_since_last_grab < 10.0 
                        else self.colors['warning'])
            else:
                color = self.colors['text']
            
            # Draw step header
            cv2.putText(panel, f"Cycle {step.step_number}",
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, color, 1)
            y_pos += 20
            
            # Draw progress bar
            bar_width = 330
            bar_height = 12
            
            # Background bar
            cv2.rectangle(panel,
                         (10, y_pos),
                         (10 + bar_width, y_pos + bar_height),
                         self.colors['text'], 1)
            
            # Progress fill
            if step.total_active_time > 0:
                progress = min(100, (step.total_active_time / 30.0) * 100)  # 30s reference
                filled_width = int(bar_width * progress/100)
                cv2.rectangle(panel,
                            (10, y_pos),
                            (10 + filled_width, y_pos + bar_height),
                            color, -1)
            
            # Status and timing information
            status = "Completed" if is_completed else "Active" if is_current else "Waiting"
            time_text = f"Time: {step.total_active_time:.1f}s ({status})"
            cv2.putText(panel, time_text,
                       (10, y_pos + 25), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, color, 1)
            
            y_pos += step_height
        
        return y_pos
    def _draw_performance_metrics(self, panel: np.ndarray, y_pos: int) -> int:
        """Draw real-time performance metrics"""
        if len(self.analytics.cycles) > 0:
            cv2.putText(panel, "Performance Metrics:",
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       self.colors['accent'], 1)
            y_pos += 25
            
            # Calculate metrics
            current_cycle_time = time.time() - self.current_cycle.start_time
            avg_cycle_time = np.mean([c.total_processing_time 
                                    for c in self.analytics.cycles])
            
            # Display metrics
            metrics = [
                f"Current Time: {current_cycle_time:.1f}s",
                f"Avg Cycle: {avg_cycle_time:.1f}s",
                f"Efficiency: {(avg_cycle_time/current_cycle_time*100):.1f}%"
            ]
            
            for metric in metrics:
                cv2.putText(panel, metric,
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           self.colors['text'], 1)
                y_pos += 20
        
        return y_pos

    def _draw_analytics_section(self, panel: np.ndarray, y_pos: int) -> int:
        """Draw production analytics summary"""
        cv2.putText(panel, "Production Analytics",
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, self.colors['accent'], 2)
        y_pos += 30
        
        # Summary statistics
        total_cycles = len(self.analytics.cycles)
        cv2.putText(panel, f"Total Cycles: {total_cycles}",
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, self.colors['text'], 1)
        
        if total_cycles > 0:
            y_pos += 25
            avg_time = np.mean([c.total_processing_time for c in self.analytics.cycles])
            cv2.putText(panel, f"Avg. Cycle Time: {avg_time:.1f}s",
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, self.colors['text'], 1)
            
            y_pos += 25
            avg_quality = np.mean([c.quality_score for c in self.analytics.cycles])
            cv2.putText(panel, f"Avg. Quality Score: {avg_quality:.1f}%",
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, self.colors['text'], 1)
        
        return y_pos
    
def run_production_tracking():
    """
    Run the production tracking system with video recording capability
    """
    try:
        # Initialize video from file instead of camera
        cap = cv2.VideoCapture('demo.mp4')
        
        # Set starting point to 25th second
        start_frame = 25 * int(cap.get(cv2.CAP_PROP_FPS))  # Convert seconds to frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Initialize tracking system
        hand_tracker = HandTracking()
        production_tracker = ProductionTracking(hand_tracker)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        recording = False
        
        print("Production Tracking System Started")
        print("Press 'q' to quit")
        print("Press 'r' to reset steps")
        print("Press 'v' to start/stop recording")
        
        last_frame = None
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Video ended or failed to read frame")
                break
            
            # Resize frame to 1280x720
            frame = cv2.resize(frame, (1280, 720))
                
            processed_frame, _ = production_tracker.process_frame(frame)
            
            if recording and out is not None:
                out.write(processed_frame)
                
            if recording:
                cv2.putText(processed_frame, "Recording...", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Production Tracking', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                production_tracker.current_step = 1
                production_tracker.active_time = 0
                production_tracker.is_step_active = False
                print("Steps reset")
            elif key == ord('v'):
                if not recording:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f'production_recording_{timestamp}.mp4'
                    out = cv2.VideoWriter(filename, fourcc, 30.0, (1280, 720))
                    recording = True
                    print(f"Recording started - {filename}")
                else:
                    out.release()
                    out = None
                    recording = False
                    print("Recording stopped")
    
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        if 'cap' in locals():
            cap.release()
        if 'out' in locals() and out is not None:
            out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_production_tracking()