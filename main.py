import cv2
import mediapipe as mp
import numpy as np
import math
import time
from math import degrees, radians
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
IMAGE_FILES = []
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
    for idx, file in enumerate(IMAGE_FILES):
        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        image = cv2.flip(cv2.imread(file), 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print handedness and draw hand landmarks on the image.
        print('Handedness:', results.multi_handedness)
        if not results.multi_hand_landmarks:
            continue
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            print('hand_landmarks:', hand_landmarks)
            print(
                f'Index finger tip coordinates: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
            )
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            cv2.imwrite(
                '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
            # Draw hand world landmarks.
            if not results.multi_hand_world_landmarks:
                continue
            for hand_world_landmarks in results.multi_hand_world_landmarks:
                mp_drawing.plot_landmarks(
                hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

def add_transparent_background(background,foreground,x_offset = None,y_offset = None):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, "Background image must have 3 channels (RGB)"
    assert fg_channels == 4, "Foreground image must have 4 channels (RGBA)"

    if x_offset is None:
        x_offset = int((bg_w - fg_w) / 2)
    if y_offset is None:
        y_offset = int((bg_h - fg_h) / 2)

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w <1 or h < 1:
        return
    
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, -x_offset)
    fg_y = max(0, -y_offset)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255.0

    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def detect_finger_gestures(hand_landmarks):
    # Lấy vị trí các landmarks quan trọng
    thumbs = [
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC],
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP],
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP],
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    ]
    
    index_fingers = [
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    ]
    
    middle_fingers = [
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ]
    
    ring_fingers = [
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP],
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ]
    
    pinky_fingers = [
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    ]
    
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    # Kiểm tra ngón tay cái (lên/xuống)
    # Ngón cái lên khi đầu ngón cái cao hơn đốt ngón cái
    thumb_up = thumbs[3].y < thumbs[2].y
    
    # Kiểm tra các ngón tay có mở hay không
    index_open = index_fingers[3].y < index_fingers[1].y
    middle_open = middle_fingers[3].y < middle_fingers[1].y
    ring_open = ring_fingers[3].y < ring_fingers[1].y
    pinky_open = pinky_fingers[3].y < pinky_fingers[1].y
    
    # Đếm số ngón tay đang mở
    open_fingers = sum([index_open, middle_open, ring_open, pinky_open])
    
    # Xác định các cử chỉ
    accelerating = thumb_up  # Ngón cái lên = tăng tốc
    braking = not thumb_up   # Ngón cái xuống = phanh
    
    # Drift (ngón trỏ mở, các ngón khác khép)
    drifting = index_open and not middle_open and not ring_open and not pinky_open
    
    # Nitro (tất cả ngón tay đều mở)
    nitro = thumb_up and index_open and middle_open and ring_open and pinky_open
    
    return {
        "thumb_up": thumb_up,
        "index_open": index_open,
        "middle_open": middle_open,
        "ring_open": ring_open,
        "pinky_open": pinky_open,
        "open_fingers": open_fingers,
        "accelerating": accelerating,
        "braking": braking,
        "drifting": drifting,
        "nitro": nitro
    }

class Car:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.position_x = width // 2  # Vị trí ban đầu X (giữa màn hình)
        self.position_y = height * 0.7  # Vị trí ban đầu Y (phía dưới màn hình)
        self.direction = 0      # Góc hướng (độ)
        self.speed = 0          # Tốc độ hiện tại
        self.max_speed = 200    # Tốc độ tối đa
        self.acceleration = 50  # Gia tốc (đơn vị/giây^2)
        self.handling = 1.0     # Độ nhạy lái xe
        self.grip = 0.9         # Hệ số bám đường (0-1)
        self.drift_factor = 0   # Hệ số trượt (0 = không trượt)
        self.score = 0          # Điểm số
        self.lap_count = 0      # Số vòng đua đã hoàn thành
        self.checkpoints = []   # Danh sách các checkpoint
        self.lap_start_time = 0 # Thời gian bắt đầu vòng đua
        self.best_lap_time = 0  # Thời gian vòng đua tốt nhất
        self.off_track = False  # Xe có đang ở ngoài đường đua không
        
        # Tạo hình ảnh xe đơn giản (tam giác)
        self.car_image = np.zeros((30, 20, 4), dtype=np.uint8)
        cv2.fillPoly(self.car_image, [np.array([[10, 0], [20, 29], [0, 29]])], (0, 0, 255, 255))
    
    def steer(self, angle):
        # Chuyển đổi góc vô lăng thành góc lái thực tế
        target_angle = -angle  # Đảo ngược để cảm giác lái tự nhiên hơn
        
        # Giới hạn góc lái
        max_steering = 45 * (1 - self.speed / self.max_speed * 0.5)
        steering_change = min(5, max(0.1, self.speed) * self.handling * 0.1)
        
        # Giới hạn góc lái trong khoảng [-max_steering, max_steering]
        target_angle = max(-max_steering, min(max_steering, target_angle))
        
        # Cập nhật hướng xe dần dần
        if abs(self.direction - target_angle) > steering_change:
            if self.direction < target_angle:
                self.direction += steering_change
            else:
                self.direction -= steering_change
        else:
            self.direction = target_angle
    
    def accelerate(self, delta_time):
        # Tăng tốc dần dần
        self.speed += self.acceleration * delta_time
        if self.speed > self.max_speed:
            self.speed = self.max_speed
    
    def brake(self, delta_time):
        # Giảm tốc nhanh hơn tăng tốc
        self.speed -= self.acceleration * 2 * delta_time
        if self.speed < 0:
            self.speed = 0
    
    def activate_nitro(self):
        # Kích hoạt nitro (tăng tốc đột ngột)
        self.speed = min(self.max_speed * 1.5, self.speed * 1.5)
    
    def drift(self):
        # Kích hoạt trượt
        self.drift_factor = 0.7
    
    def update(self, delta_time, track_image=None):
        # Tính toán thành phần vận tốc dựa trên hướng xe
        velocity_x = self.speed * math.sin(radians(self.direction))
        velocity_y = -self.speed * math.cos(radians(self.direction))  # Âm vì trục y đi xuống
        
        # Áp dụng lực ma sát và trượt
        effective_grip = self.grip * (1 - self.drift_factor)
        
        # Cập nhật vị trí
        self.position_x += velocity_x * delta_time
        self.position_y += velocity_y * delta_time
        
        # Kiểm tra va chạm với biên màn hình
        self.position_x = max(0, min(self.position_x, self.width))
        self.position_y = max(0, min(self.position_y, self.height))
        
        # Kiểm tra va chạm với đường đua nếu có track_image
        if track_image is not None:
            try:
                # Lấy màu tại vị trí hiện tại của xe
                pixel_value = track_image[int(self.position_y), int(self.position_x)]
                # Nếu màu là màu đen (ngoài đường đua), giảm tốc độ
                if np.all(pixel_value < 50):  # Ngưỡng màu đen
                    self.speed *= 0.9
                    self.off_track = True
                else:
                    self.off_track = False
            except IndexError:
                # Nếu vị trí nằm ngoài hình ảnh
                self.speed *= 0.5
        
        # Giảm dần hiệu ứng drift
        self.drift_factor *= 0.95
        
        # Cập nhật điểm số
        if self.speed > 10:  # Chỉ tính điểm khi đang di chuyển
            self.score += int(self.speed * delta_time * 0.1)
        
        # Phạt điểm nếu ra khỏi đường
        if self.off_track:
            self.score = max(0, self.score - int(10 * delta_time))
    
    def render(self, frame):
        # Xoay và vẽ xe lên frame
        rotated_car = rotate_image(self.car_image, self.direction)
        car_x = int(self.position_x - rotated_car.shape[1] / 2)
        car_y = int(self.position_y - rotated_car.shape[0] / 2)
        add_transparent_background(frame, rotated_car, car_x, car_y)
        
        return frame

class Track:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.track_image = self.create_track()
        self.checkpoints = self.create_checkpoints()
    
    def create_track(self):
        # Tạo hình ảnh đường đua đơn giản
        track_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Tạo đường đua hình oval
        cv2.ellipse(track_img, (self.width//2, self.height//2), (self.width//3, self.height//3), 
                    0, 0, 360, (200, 200, 200), -1)
        cv2.ellipse(track_img, (self.width//2, self.height//2), (self.width//4, self.height//4), 
                    0, 0, 360, (50, 50, 50), -1)
        
        # Vẽ vạch xuất phát
        cv2.rectangle(track_img, (self.width//2 - 50, self.height//2 + self.height//3 - 10),
                     (self.width//2 + 50, self.height//2 + self.height//3 + 10), (0, 0, 255), -1)
        
        return track_img
    
    def create_checkpoints(self):
        # Tạo các checkpoint trên đường đua
        checkpoints = []
        # Thêm checkpoint ở vạch xuất phát
        checkpoints.append({
            "position": (self.width//2, self.height//2 + self.height//3),
            "passed": False
        })
        # Thêm checkpoint ở trên cùng
        checkpoints.append({
            "position": (self.width//2, self.height//2 - self.height//3),
            "passed": False
        })
        # Thêm checkpoint ở bên trái
        checkpoints.append({
            "position": (self.width//2 - self.width//3, self.height//2),
            "passed": False
        })
        # Thêm checkpoint ở bên phải
        checkpoints.append({
            "position": (self.width//2 + self.width//3, self.height//2),
            "passed": False
        })
        
        return checkpoints
    
    def reset_checkpoints(self):
        for checkpoint in self.checkpoints:
            checkpoint["passed"] = False
    
    def check_checkpoint(self, car_x, car_y):
        for i, checkpoint in enumerate(self.checkpoints):
            # Khoảng cách từ xe đến checkpoint
            distance = math.sqrt((car_x - checkpoint["position"][0])**2 + (car_y - checkpoint["position"][1])**2)
            
            # Nếu xe đủ gần checkpoint và checkpoint chưa được đi qua
            if distance < 30 and not checkpoint["passed"]:
                checkpoint["passed"] = True
                # Kiểm tra xem đã qua hết checkpoint chưa
                if i == 0 and all(cp["passed"] for cp in self.checkpoints[1:]):
                    # Hoàn thành một vòng đua
                    self.reset_checkpoints()
                    return True
                return False
        
        return False
    
    def render(self, frame):
        # Vẽ đường đua lên frame
        frame_copy = frame.copy()
        track_resized = cv2.resize(self.track_image, (frame.shape[1], frame.shape[0]))
        cv2.addWeighted(frame_copy, 0.7, track_resized, 0.3, 0, frame)
        
        # Vẽ các checkpoint
        for checkpoint in self.checkpoints:
            color = (0, 255, 0) if checkpoint["passed"] else (0, 0, 255)
            cv2.circle(frame, checkpoint["position"], 10, color, -1)
        
        return frame

class Game:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.car = Car(width, height)
        self.track = Track(width, height)
        self.start_time = 0
        self.game_time = 0
        self.game_state = "READY"  # READY, COUNTDOWN, RACING, FINISHED
        self.countdown = 3
        
    def start(self):
        self.game_state = "COUNTDOWN"
        self.countdown = 3
    
    def update(self, delta_time, steering_angle=None, hand_gestures=None):
        if self.game_state == "READY":
            # Chờ người chơi bắt đầu
            pass
        
        elif self.game_state == "COUNTDOWN":
            # Đếm ngược
            self.countdown -= delta_time
            if self.countdown <= 0:
                self.game_state = "RACING"
                self.start_time = time.time()
                self.car.lap_start_time = time.time()
        
        elif self.game_state == "RACING":
            # Tính thời gian
            self.game_time = time.time() - self.start_time
            
            # Cập nhật tốc độ dựa vào cử chỉ tay
            if hand_gestures:
                if hand_gestures.get("accelerating", False):
                    self.car.accelerate(delta_time)
                if hand_gestures.get("braking", False):
                    self.car.brake(delta_time)
                if hand_gestures.get("drifting", False):
                    self.car.drift()
                if hand_gestures.get("nitro", False):
                    self.car.activate_nitro()
            
            # Cập nhật hướng dựa vào góc lái
            if steering_angle is not None:
                self.car.steer(steering_angle)
            
            # Cập nhật vị trí xe
            self.car.update(delta_time, self.track.track_image)
            
            # Kiểm tra checkpoint
            if self.track.check_checkpoint(self.car.position_x, self.car.position_y):
                self.car.lap_count += 1
                lap_time = time.time() - self.car.lap_start_time
                
                # Cập nhật thời gian vòng đua tốt nhất
                if lap_time < self.car.best_lap_time or self.car.best_lap_time == 0:
                    self.car.best_lap_time = lap_time
                
                # Bắt đầu tính thời gian vòng đua mới
                self.car.lap_start_time = time.time()
                
                # Thưởng điểm
                self.car.score += 1000
                
                # Kiểm tra kết thúc đua (3 vòng)
                if self.car.lap_count >= 3:
                    self.game_state = "FINISHED"
    
    def render(self, frame):
        # Vẽ đường đua
        frame = self.track.render(frame)
        
        # Vẽ xe
        frame = self.car.render(frame)
        
        # Hiển thị UI
        if self.game_state == "READY":
            cv2.putText(frame, "SAN SANG!", (self.width//2 - 100, self.height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Dua 2 tay vao vi tri vo lang de bat dau", 
                       (self.width//2 - 250, self.height//2 + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        elif self.game_state == "COUNTDOWN":
            cv2.putText(frame, str(int(self.countdown) + 1), (self.width//2 - 25, self.height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
        
        elif self.game_state == "RACING" or self.game_state == "FINISHED":
            # Hiển thị tốc độ
            cv2.putText(frame, f"Toc do: {int(self.car.speed)} km/h", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Hiển thị thời gian
            minutes = int(self.game_time) // 60
            seconds = int(self.game_time) % 60
            cv2.putText(frame, f"Thoi gian: {minutes:02d}:{seconds:02d}", (50, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Hiển thị vòng đua
            cv2.putText(frame, f"Vong dua: {self.car.lap_count}/3", (50, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Hiển thị điểm số
            cv2.putText(frame, f"Diem so: {self.car.score}", (self.width - 200, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Hiển thị thời gian vòng đua tốt nhất
            if self.car.best_lap_time > 0:
                best_min = int(self.car.best_lap_time) // 60
                best_sec = int(self.car.best_lap_time) % 60
                best_ms = int((self.car.best_lap_time % 1) * 100)
                cv2.putText(frame, f"Lap tot nhat: {best_min:02d}:{best_sec:02d}.{best_ms:02d}", 
                           (self.width - 350, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        if self.game_state == "FINISHED":
            cv2.putText(frame, "HOAN THANH!", (self.width//2 - 150, self.height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.putText(frame, f"Tong diem: {self.car.score}", 
                       (self.width//2 - 100, self.height//2 + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame

# Load vô lăng
wheel_image = cv2.imread('volang.png', cv2.IMREAD_UNCHANGED)

# Khởi tạo webcam
cap = cv2.VideoCapture(0)
success, temp_frame = cap.read()
if success:
    frame_height, frame_width, _ = temp_frame.shape
    game = Game(frame_width, frame_height)
else:
    print("Không thể mở webcam")
    exit()

# Biến đếm thời gian
prev_time = time.time()

# Khởi tạo MediaPipe Hands
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Tính delta time
        current_time = time.time()
        delta_time = current_time - prev_time
        prev_time = current_time

        # Xử lý khung hình để nhận diện tay
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Chuyển về BGR để hiển thị
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        height, width, _ = image.shape

        # Biến lưu thông tin cử chỉ tay
        hand_gestures = None
        steering_angle = None

        if results.multi_hand_landmarks:
            hand_centers = []
            
            # Xử lý từng tay được phát hiện
            for hand_landmarks in results.multi_hand_landmarks:
                # Vẽ landmarks của tay
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                # Lấy vị trí trung tâm lòng bàn tay
                hand_centers.append(
                    [int(hand_landmarks.landmark[9].x * width), int(hand_landmarks.landmark[9].y * height)])
                
                # Nếu chỉ phát hiện một tay, phân tích cử chỉ ngón tay
                if len(results.multi_hand_landmarks) == 1:
                    hand_gestures = detect_finger_gestures(hand_landmarks)
            
            # Nếu phát hiện 2 bàn tay (vô lăng)
            if len(hand_centers) == 2:
                center_x = int((hand_centers[0][0] + hand_centers[1][0]) / 2)
                center_y = int((hand_centers[0][1] + hand_centers[1][1]) / 2)
                radius = int(math.sqrt((hand_centers[0][0] - hand_centers[1][0])**2 + 
                                     (hand_centers[0][1] - hand_centers[1][1])**2) / 2)
                
                # Tính góc giữa 2 bàn tay
                steering_angle = degrees(math.atan2(hand_centers[1][1] - hand_centers[0][1], 
                                                   hand_centers[1][0] - hand_centers[0][0]))
                
                # Hiển thị vô lăng
                add_transparent_background(image, 
                                        cv2.resize(rotate_image(wheel_image, 180-steering_angle), 
                                                  (2*radius, 2*radius)), 
                                        int(center_x - radius), int(center_y - radius))
                
                # Nếu game đang ở trạng thái READY và người chơi đã đặt tay vào vị trí vô lăng
                if game.game_state == "READY" and radius > 30:
                    game.start()

        # Cập nhật trạng thái game
        game.update(delta_time, steering_angle, hand_gestures)
        
        # Render game
        image = game.render(image)

        # Hiển thị hình ảnh
        cv2.imshow('MediaPipe Hands Racing Game', cv2.flip(image, 1))
        
        # Nhấn ESC để thoát
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
