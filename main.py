import cv2
import mediapipe as mp
import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *

class HandTracking3D:
    def __init__(self, width=1280, height=480):
        # Previous initialization code remains the same
        self.width = width
        self.height = height
        pygame.init()
        self.display = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.webcam_surface = pygame.Surface((640, 480))
        
        # Modified initialization for scaling
        self.cube_pos = [0, 0, -5]
        self.selected = False
        self.initial_hand_pos = None
        self.initial_cube_pos = None
        self.cube_rotation = [0, 0, 0]
        self.cube_scale = [1.0, 1.0, 1.0]  # Initialize as float
        self.selected_mode = 'position'
        self.initial_rotation = None
        self.initial_scale = None
        self.previous_scale_distance = None
        self.base_scale = 1.0  # Add base scale for reference
        
        self.setup_gl()

    def setup_gl(self):
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.width/2/self.height), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        glClearColor(0.2, 0.2, 0.2, 1)

    def check_three_finger_pinch(self, hand_landmarks):
        # Get landmarks for thumb, index, and middle fingers
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        
        # Calculate distances between all three fingers
        thumb_index_dist = np.sqrt(
            (thumb_tip.x - index_tip.x)**2 +
            (thumb_tip.y - index_tip.y)**2 +
            (thumb_tip.z - index_tip.z)**2
        )
        
        thumb_middle_dist = np.sqrt(
            (thumb_tip.x - middle_tip.x)**2 +
            (thumb_tip.y - middle_tip.y)**2 +
            (thumb_tip.z - middle_tip.z)**2
        )
        
        index_middle_dist = np.sqrt(
            (index_tip.x - middle_tip.x)**2 +
            (index_tip.y - middle_tip.y)**2 +
            (index_tip.z - middle_tip.z)**2
        )
        
        # All three fingers should be close together
        threshold = 0.1
        return all(dist < threshold for dist in [thumb_index_dist, thumb_middle_dist, index_middle_dist])

    def check_index_pinch(self, hand_landmarks):
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        
        distance = np.sqrt(
            (thumb_tip.x - index_tip.x)**2 +
            (thumb_tip.y - index_tip.y)**2 +
            (thumb_tip.z - index_tip.z)**2
        )
        
        return distance < 0.1

    def get_hand_position(self, hand_landmarks):
        index_tip = hand_landmarks.landmark[8]
        return np.array([
            (index_tip.x - 0.5) * 2,
            -(index_tip.y - 0.5) * 2,
            index_tip.z
        ])

    def calculate_scale_from_hands(self, hand_landmarks1, hand_landmarks2):
        # Get index finger tips from both hands
        index1 = hand_landmarks1.landmark[8]
        index2 = hand_landmarks2.landmark[8]
        
        # Calculate distance between index fingers in 2D space
        current_distance = np.sqrt(
            (index1.x - index2.x)**2 +
            (index1.y - index2.y)**2
        )
        
        if self.previous_scale_distance is None:
            self.previous_scale_distance = current_distance
            self.base_scale = self.cube_scale[0]  # Store current scale as base
            return 1.0
        
        # Calculate relative change in distance
        scale_change = (current_distance / self.previous_scale_distance)
        
        # Make scaling more dramatic
        scale_change = 1.0 + (scale_change - 1.0)
        
        # Update previous distance
        self.previous_scale_distance = current_distance
        
        return scale_change

    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            return False

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        # Draw hand tracking for all detected hands
        self.draw_hand_tracking(frame, results)

        if results.multi_hand_landmarks:
            primary_hand = results.multi_hand_landmarks[0]
            current_hand_pos = self.get_hand_position(primary_hand)

            # Three finger pinch for rotation
            if self.check_three_finger_pinch(primary_hand):
                if not self.selected or self.selected_mode != 'rotation':
                    self.selected = True
                    self.selected_mode = 'rotation'
                    self.initial_hand_pos = current_hand_pos
                    self.initial_rotation = np.array(self.cube_rotation)
                else:
                    delta = current_hand_pos - self.initial_hand_pos
                    self.cube_rotation[1] = self.initial_rotation[1] + delta[0] * 180
                    self.cube_rotation[0] = self.initial_rotation[0] + delta[1] * 180

            # Single hand position control
            elif self.check_index_pinch(primary_hand) and len(results.multi_hand_landmarks) == 1:
                if not self.selected or self.selected_mode != 'position':
                    self.selected = True
                    self.selected_mode = 'position'
                    self.initial_hand_pos = current_hand_pos
                    self.initial_cube_pos = np.array(self.cube_pos)
                else:
                    delta = current_hand_pos - self.initial_hand_pos
                    self.cube_pos = self.initial_cube_pos + delta * 5

            # Two-handed scaling
            if len(results.multi_hand_landmarks) == 2:
                hand1 = results.multi_hand_landmarks[0]
                hand2 = results.multi_hand_landmarks[1]
                
                if self.check_index_pinch(hand1) and self.check_index_pinch(hand2):
                    if self.selected_mode != 'scale':
                        self.selected = True
                        self.selected_mode = 'scale'
                        self.previous_scale_distance = None
                    
                    scale_factor = self.calculate_scale_from_hands(hand1, hand2)
                    new_scale = np.array(self.cube_scale) * scale_factor
                    # Limit the scale range
                    if all(0.1 <= s <= 5.0 for s in new_scale):
                        self.cube_scale = new_scale.tolist()  # Convert back to list
            
            # Reset scaling when not using two hands
            elif self.selected_mode == 'scale':
                self.selected = False
                self.selected_mode = 'none'
                self.previous_scale_distance = None

            # Reset if no gestures are active
            if not any([
                self.check_three_finger_pinch(primary_hand),
                (len(results.multi_hand_landmarks) == 1 and self.check_index_pinch(primary_hand)),
                (len(results.multi_hand_landmarks) == 2 and 
                 self.check_index_pinch(results.multi_hand_landmarks[0]) and 
                 self.check_index_pinch(results.multi_hand_landmarks[1]))
            ]):
                self.selected = False
                self.selected_mode = 'none'

        # Rest of the update code (rendering) remains the same
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)
        frame = pygame.transform.flip(frame, True, True)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Draw webcam feed
        glViewport(0, 0, self.width//2, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, 1, 0, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glColor3f(1.0, 1.0, 1.0)
        glEnable(GL_TEXTURE_2D)
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.get_width(), frame.get_height(),
                    0, GL_RGB, GL_UNSIGNED_BYTE, pygame.image.tostring(frame, 'RGB', True))
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(0, 0)
        glTexCoord2f(1, 1); glVertex2f(1, 0)
        glTexCoord2f(1, 0); glVertex2f(1, 1)
        glTexCoord2f(0, 0); glVertex2f(0, 1)
        glEnd()

        glDeleteTextures([texture_id])
        glDisable(GL_TEXTURE_2D)

        # Draw 3D scene
        glViewport(self.width//2, 0, self.width//2, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.width/2/self.height), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        self.draw_cube()
        pygame.display.flip()
        
        return True

    def draw_cube(self):
        glLoadIdentity()
        glTranslatef(self.cube_pos[0], self.cube_pos[1], self.cube_pos[2])

        # Apply rotations
        glRotatef(self.cube_rotation[0], 1, 0, 0)  # X-axis rotation
        glRotatef(self.cube_rotation[1], 0, 1, 0)  # Y-axis rotation
        glRotatef(self.cube_rotation[2], 0, 0, 1)  # Z-axis rotation
        
        # Apply scaling
        glScalef(self.cube_scale[0], self.cube_scale[1], self.cube_scale[2])
        
        # Draw edges first (black)
        glColor3f(0.0, 0.0, 0.0)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        # Front face
        glVertex3f(-0.5, -0.5, 0.5); glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(0.5, -0.5, 0.5); glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(0.5, 0.5, 0.5); glVertex3f(-0.5, 0.5, 0.5)
        glVertex3f(-0.5, 0.5, 0.5); glVertex3f(-0.5, -0.5, 0.5)
        # Back face
        glVertex3f(-0.5, -0.5, -0.5); glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(0.5, -0.5, -0.5); glVertex3f(0.5, 0.5, -0.5)
        glVertex3f(0.5, 0.5, -0.5); glVertex3f(-0.5, 0.5, -0.5)
        glVertex3f(-0.5, 0.5, -0.5); glVertex3f(-0.5, -0.5, -0.5)
        # Connecting lines
        glVertex3f(-0.5, -0.5, 0.5); glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(0.5, -0.5, 0.5); glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(0.5, 0.5, 0.5); glVertex3f(0.5, 0.5, -0.5)
        glVertex3f(-0.5, 0.5, 0.5); glVertex3f(-0.5, 0.5, -0.5)
        glEnd()

        # Draw faces
        if self.selected:
            glColor4f(1.0, 0.0, 0.0, 0.5)  # Semi-transparent red when selected
        else:
            glColor4f(0.0, 1.0, 0.0, 0.5)  # Semi-transparent green when not selected

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glBegin(GL_QUADS)
        # Front face
        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        # Back face
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(-0.5, 0.5, -0.5)
        glVertex3f(0.5, 0.5, -0.5)
        glVertex3f(0.5, -0.5, -0.5)
        # Top face
        glVertex3f(-0.5, 0.5, -0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(0.5, 0.5, -0.5)
        # Bottom face
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(-0.5, -0.5, 0.5)
        # Right face
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(0.5, 0.5, -0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(0.5, -0.5, 0.5)
        # Left face
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        glVertex3f(-0.5, 0.5, -0.5)
        glEnd()
        
        glDisable(GL_BLEND)

    def draw_hand_tracking(self, frame, results):
        # Draw all detected hands
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(255,255,255), thickness=2)
                )
        
        # Draw mode and debug information
        cv2.putText(frame, f"Mode: {self.selected_mode}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Scale: {self.cube_scale[0]:.2f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if self.selected_mode == 'scale':
            cv2.putText(frame, "Scaling Active", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
                cv2.putText(frame, "Two Hands Detected", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            if not self.update():
                break

        self.cap.release()
        pygame.quit()

if __name__ == "__main__":
    app = HandTracking3D()
    app.run()