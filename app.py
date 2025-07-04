from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import base64
import io
import copy
import threading
import time
import json
from PIL import Image
import kociemba

app = Flask(__name__)

# Global variables
current_cube_state = {}
camera_feed = None
solving_in_progress = False
solution_moves = []
current_move_index = 0

def classify_hue(h, s, v):
    """Classify HSV values to cube face colors"""
    if h >= 5 and h <= 36 and s >= 9 and s <= 60 and v >= 45 and v <= 179:
        return "W"
    elif h >= 0 and h <= 25 and s >= 156 and s <= 232 and v >= 82 and v <= 143:
        return "R"
    elif h >= 28 and h <= 39 and s >= 146 and s <= 255 and v >= 132 and v <= 194:
        return "Y"
    elif h >= 42 and h <= 160 and s >= 133 and s <= 255 and v >= 97 and v <= 190:
        return "G"
    elif h >= 55 and h <= 121 and s >= 129 and s <= 255 and v >= 26 and v <= 84:
        return "B"
    elif h >= 1 and h <= 85 and s >= 211 and s <= 248 and v >= 75 and v <= 148:
        return "O"
    else:
        return "O"

def rotate_face(face, turns=1):
    """Rotate a face of the cube"""
    for _ in range(turns % 4):
        face[:] = [
            face[6], face[3], face[0],
            face[7], face[4], face[1],
            face[8], face[5], face[2]
        ]
    return face

def cycle_edges(state, faces, indices, turns=1):
    """Cycle edges during cube moves"""
    for _ in range(turns % 4):
        tmp = [state[faces[-1]][i] for i in indices[-1]]
        for i in reversed(range(1, 4)):
            for j in range(3):
                state[faces[i]][indices[i][j]] = state[faces[i - 1]][indices[i - 1][j]]
        for j in range(3):
            state[faces[0]][indices[0][j]] = tmp[j]

def apply_move(state, move):
    """Apply a move to the cube state"""
    face = move[0]
    modifier = move[1:] if len(move) > 1 else ''
    turns = {'': 1, "'": 3, '2': 2}[modifier]
    state = copy.deepcopy(state)
    rotate_face(state[face], turns)
    
    if face == 'U':
        cycle_edges(state, ['B', 'R', 'F', 'L'], [[0,1,2]]*4, turns)
    elif face == 'D':
        cycle_edges(state, ['F', 'R', 'B', 'L'], [[6,7,8]]*4, turns)
    elif face == 'F':
        cycle_edges(state, ['U', 'R', 'D', 'L'], [[6,7,8], [0,3,6], [2,1,0], [8,5,2]], turns)
    elif face == 'B':
        cycle_edges(state, ['U', 'L', 'D', 'R'], [[2,1,0], [0,3,6], [6,7,8], [8,5,2]], turns)
    elif face == 'L':
        cycle_edges(state, ['U', 'F', 'D', 'B'], [[0,3,6]]*3 + [[8,5,2]], turns)
    elif face == 'R':
        cycle_edges(state, ['U', 'B', 'D', 'F'], [[8,5,2], [0,3,6], [8,5,2], [8,5,2]], turns)
    return state

def get_camera_frame():
    """Get frame from camera"""
    global camera_feed
    if camera_feed is None:
        camera_feed = cv2.VideoCapture(0)
    
    ret, frame = camera_feed.read()
    if ret:
        frame = cv2.resize(frame, (750, 640))
        return frame
    return None

def scan_cube_face(frame):
    """Scan a cube face from camera frame"""
    GRID_SIZE = 3
    SPACING = 160
    
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    center_x, center_y = width // 2, height // 2
    
    current_face = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            x = center_x + (j - 1) * SPACING
            y = center_y + (i - 1) * SPACING + 50
            hsv_pixel = hsv[y, x]
            h, s, v = hsv_pixel
            color = classify_hue(h, s, v)
            current_face.append(color)
            
            # Draw scanning points
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(frame, color, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return current_face, frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera_feed')
def camera_feed_route():
    """Video streaming route"""
    def generate():
        while True:
            frame = get_camera_frame()
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/scan_face', methods=['POST'])
def scan_face():
    """Scan a specific face of the cube"""
    face_name = request.json.get('face')
    
    frame = get_camera_frame()
    if frame is None:
        return jsonify({'error': 'Camera not available'}), 500
    
    face_colors, annotated_frame = scan_cube_face(frame)
    current_cube_state[face_name] = face_colors
    
    return jsonify({
        'face': face_name,
        'colors': face_colors,
        'scanned_faces': list(current_cube_state.keys())
    })

@app.route('/solve_cube', methods=['POST'])
def solve_cube():
    """Solve the cube using Kociemba algorithm"""
    global solution_moves, current_move_index, solving_in_progress
    
    if len(current_cube_state) != 6:
        return jsonify({'error': 'Please scan all 6 faces first'}), 400
    
    try:
        # Build cube string for Kociemba
        face_order = ['U', 'R', 'F', 'D', 'L', 'B']
        color_to_face = {current_cube_state[face][4]: face for face in face_order}
        cube_string = ''.join(color_to_face.get(color, '?') for face in face_order for color in current_cube_state[face])
        
        # Solve using Kociemba
        solution = kociemba.solve(cube_string)
        solution_moves = solution.strip().split() if solution else []
        current_move_index = 0
        solving_in_progress = True
        
        return jsonify({
            'solution': solution,
            'moves': solution_moves,
            'total_moves': len(solution_moves)
        })
        
    except Exception as e:
        return jsonify({'error': f'Could not solve cube: {str(e)}'}), 500

@app.route('/get_cube_state')
def get_cube_state():
    """Get current cube state"""
    return jsonify({
        'cube_state': current_cube_state,
        'scanned_faces': list(current_cube_state.keys())
    })

@app.route('/reset_cube')
def reset_cube():
    """Reset cube state"""
    global current_cube_state, solution_moves, current_move_index, solving_in_progress
    current_cube_state = {}
    solution_moves = []
    current_move_index = 0
    solving_in_progress = False
    return jsonify({'message': 'Cube reset successfully'})

@app.route('/next_move')
def next_move():
    """Get next move in solution"""
    global current_move_index
    if current_move_index < len(solution_moves):
        move = solution_moves[current_move_index]
        current_move_index += 1
        return jsonify({
            'move': move,
            'step': current_move_index,
            'total_steps': len(solution_moves),
            'completed': current_move_index >= len(solution_moves)
        })
    return jsonify({'completed': True})

if __name__ == '__main__':
    app.run(debug=True, port=5000)