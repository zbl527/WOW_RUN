import os
import time
import pytesseract
from PIL import Image,ImageEnhance
import tkinter as tk
import tkinter.font as tkFont
import math
import win32gui
import win32ui
import win32con
import json
import networkx as nx
import numpy as np
from itertools import combinations
from sympy import symbols, Eq, solve
pytesseract.pytesseract.tesseract_cmd = os.path.join(os.getcwd(), "Tesseract-OCR", "tesseract.exe")
def find_window(title):
    hwnd = win32gui.FindWindow(None, title)
    return hwnd
hwnd = find_window('魔兽世界')
root = tk.Tk()
root.attributes("-topmost", True)
root.title(f"WOW!") 
fontStyle = tkFont.Font(family="Lucida Grande", size=14)
canvas = tk.Canvas(root, width=500, height=500, bg="white")
canvas.pack()
canvas.config(scrollregion=(0, 0, 500, 500))
red_lines = []
x_data=[]
y_data=[]
path=[]
x_coords = [0]
y_coords = [500]  
now_point = (0, 0) 
end_point = None
prev_x, prev_y = None, None
target_x, target_y = None, None
target_x0, target_y0=None, None
target_index = 0
current_angle = 0
no_change_threshold = 3
run_key = 0x57 # W键
jump_key = 0x20 # 空格键
left_turn_key = 0x51  # Q键
right_turn_key = 0x45  # E键
def load_data(filename="way.json"):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None
data = load_data()
if data is not None:
    x_data = data.get('x', [])
    y_data = data.get('y', [])
def draw_point(x, y):
    scale_factor = 500 / 100
    canvas_x = x * scale_factor
    canvas_y = y * scale_factor
    if len(x_coords) > 1:
        canvas.create_oval(canvas_x - 2, canvas_y - 2, canvas_x + 2, canvas_y + 2, fill="green",outline="green", width=1)
    x_coords.append(canvas_x)
    y_coords.append(canvas_y)
for x, y in zip(x_data, y_data):
    draw_point(x, y)

def on_right_click(event):
    global now_point,end_point, path, red_lines,target_index
    target_index = 0
    try:
        def distance(p1, p2):
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        G = nx.Graph()
        for x, y in zip(x_data, y_data):
            G.add_node((x, y))

        for (x1, y1), (x2, y2) in combinations(zip(x_data, y_data), 2):
            dist = distance((x1, y1), (x2, y2))
            if dist <= 0.5: #点连成线的路径距离
                G.add_edge((x1, y1), (x2, y2), weight=dist)
        def find_closest_node(G, point):
            closest_node = min(G.nodes, key=lambda node: distance(node, point))
            return closest_node
        end_point = (event.x / 5, event.y / 5)
        closest_now = find_closest_node(G, now_point)
        closest_end = find_closest_node(G, end_point)
        def douglas_peucker(points, epsilon):
            def find_farthest_point_index(pts):
                max_distance = 0
                index = 0
                for i in range(1, len(pts) - 1):
                    distance = point_line_distance(pts[i], pts[0], pts[-1])
                    if distance > max_distance:
                        max_distance = distance
                        index = i
                return index, max_distance
            def point_line_distance(point, start, end):
                if start == end:
                    return ((point[0] - start[0]) ** 2 + (point[1] - start[1]) ** 2) ** 0.5
                else:
                    n = abs((end[0] - start[0]) * (start[1] - point[1]) - (start[0] - point[0]) * (end[1] - start[1]))
                    d = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
                    return n / d
            index, distance = find_farthest_point_index(points)
            if distance > epsilon:
                left = douglas_peucker(points[:index+1], epsilon)
                right = douglas_peucker(points[index:], epsilon)
                return left[:-1] + right
            else:
                return [points[0], points[-1]]
        path = douglas_peucker(nx.astar_path(G, closest_now, closest_end, heuristic=distance, weight='weight'),0) #路径简化程度0-1
        print(path)
        for line in red_lines:
            canvas.delete(line)
        red_lines.clear()  
        for i in range(len(path) - 1):
            start_point = path[i]
            end_point = path[i + 1]
            canvas_start_x, canvas_start_y = start_point[0] * 500 / 100, start_point[1] * 500 / 100
            canvas_end_x, canvas_end_y = end_point[0] * 500 / 100, end_point[1] * 500 / 100
            line_id = canvas.create_line(canvas_start_x, canvas_start_y, canvas_end_x, canvas_end_y, fill="red", width=2)
            red_lines.append(line_id)  
    except:
        pass

def capture_window(hwnd, capture_ratio=(74, 0.6, 9, 1.8)):
    window_dc = win32gui.GetWindowDC(hwnd)
    dc = win32ui.CreateDCFromHandle(window_dc)
    compatible_dc = dc.CreateCompatibleDC()
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    w = right - left
    h = bottom - top
    capture_x = int(w * capture_ratio[0] / 100)
    capture_y = int(h * capture_ratio[1] / 100)
    capture_w = int(w * capture_ratio[2] / 100)
    capture_h = int(h * capture_ratio[3] / 100)
    bitmap = win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(dc, capture_w, capture_h)
    compatible_dc.SelectObject(bitmap)
    compatible_dc.BitBlt((0, 0), (capture_w, capture_h), dc, (capture_x, capture_y), win32con.SRCCOPY)
    bmpinfo = bitmap.GetInfo()
    bmpstr = bitmap.GetBitmapBits(True)
    image = Image.frombuffer(
        'RGB',
        (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
        bmpstr, 'raw', 'BGRX', 0, 1)
    win32gui.DeleteObject(bitmap.GetHandle())
    compatible_dc.DeleteDC()
    dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, window_dc)
    return image

def get_window_coordinates():
    screenshot = capture_window(hwnd)
    gray_image = screenshot.convert('L')
    enhancer = ImageEnhance.Contrast(gray_image)
    contrasted_image = enhancer.enhance(2)
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.X'
    text = pytesseract.image_to_string(contrasted_image, config=custom_config)
    info = text.split('X')
    def safe_float_convert(value, default=0.0):
        try:
            return float(value)
        except ValueError:
            return default
    def add_data(x_data, y_data, current_x, current_y):
        global now_point
        def convert_value(value):
            if value > 100:
                value_str = str(value)
                new_value = float(value_str[:2] + '.' + value_str[2:3])
            else:
                new_value = value
            return new_value
        current_x = convert_value(current_x)
        current_y = convert_value(current_y)
        if len(x_data) >= 5 and len(y_data) >= 5:
            avg_x = sum(x_data[-5:]) / 5
            avg_y = sum(y_data[-5:]) / 5
            if all(x == x_data[-1] for x in x_data[-5:]) and all(y == y_data[-1] for y in y_data[-5:]):
                current_x,current_y = current_x,current_y
            elif abs(current_x - avg_x) > 3 or abs(current_y - avg_y) > 3: #OCR识别偏移量校准
                current_x,current_y=x_data[-1], y_data[-1]
                print("偏移量校准")
        x_data.append(current_x)
        y_data.append(current_y)
        now_point=(current_x,current_y)
        return current_x, current_y
    current_x,current_y = safe_float_convert(info[0]), safe_float_convert(info[1])
    current_x,current_y = add_data(x_data, y_data,current_x, current_y)
    print(f"{current_x},{current_y}")
    return current_x,current_y

def move_to_target(current_x, current_y, target_x, target_y, prev_x, prev_y):
    global no_change_counter,smart_turn, first_loop_after_reaching_target
    current_angle = 0 
    def calculate_distance(x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def calculate_angle(x1, y1, x2, y2):
        return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 360

    def calculate_time_to_sleep(angle_diff):
        if angle_diff > 180:
            angle_diff = 360 - angle_diff 
        time_to_sleep = angle_diff / 180
        return time_to_sleep
    
    def target_way(current_x, current_y, target_x, target_y, target_x0, target_y0):
        m = (target_y0 - target_y) / (target_x0 - target_x)
        d = abs(m * current_x - current_y + (target_y - m * target_x)) / (m**2 + 1)**0.5
        if d >= 0.2: #路径宽度0-1
            m2 = -1 / m
            x_intersect = (m2 * current_x - current_y + target_y - m * target_x) / (m2 - m)
            y_intersect = m * (x_intersect - target_x) + target_y
            target_x, target_y = x_intersect, y_intersect
            smart_turn = 0.7
        else:
            smart_turn = 0.2 #转向敏感度0-1
        return target_x, target_y, smart_turn

    if calculate_distance(current_x, current_y, target_x, target_y) <= 0.2: #经过坐标敏感范围
        root.title(f"经过结点")
        first_loop_after_reaching_target = True
        return None, None, True 
    try:
        target_x0, target_y0 = path[target_index-1]
        target_x, target_y,smart_turn = target_way(current_x, current_y, target_x, target_y, target_x0, target_y0)
    except:
        pass

    if prev_x is not None and prev_y is not None:
        delta_x = current_x - prev_x
        delta_y = current_y - prev_y
        if abs(delta_x) < 0.05 and abs(delta_y) < 0.05:
            no_change_counter += 1
        else:
            no_change_counter = 0
        if no_change_counter >= no_change_threshold:
            win32gui.PostMessage(hwnd, win32con.WM_KEYDOWN, jump_key, 0)
            time.sleep(1)
            win32gui.PostMessage(hwnd, win32con.WM_KEYUP, jump_key, 0)
            no_change_counter = 0
        current_angle = calculate_angle(prev_x, prev_y, current_x, current_y)
    if first_loop_after_reaching_target:
        smart_turn = 0
        first_loop_after_reaching_target = False
    target_angle = calculate_angle(current_x, current_y, target_x, target_y)
    angle_diff = (target_angle - current_angle + 360) % 360
    time_to_sleep = calculate_time_to_sleep(angle_diff)
    time_to_sleep = min(time_to_sleep, smart_turn)
    if angle_diff != 0:
        if angle_diff < 180:
            win32gui.PostMessage(hwnd, win32con.WM_KEYDOWN, right_turn_key, 0)
            time.sleep(time_to_sleep)
            win32gui.PostMessage(hwnd, win32con.WM_KEYUP, right_turn_key, 0)
        else:
            win32gui.PostMessage(hwnd, win32con.WM_KEYDOWN, left_turn_key, 0)
            time.sleep(time_to_sleep)
            win32gui.PostMessage(hwnd, win32con.WM_KEYUP, left_turn_key, 0)
        win32gui.PostMessage(hwnd, win32con.WM_KEYDOWN, run_key, 0)
    root.title(f"({current_x}, {current_y}) > ({target_x}, {target_y})") 
    return current_x, current_y, False

def update_position():
    global prev_x, prev_y, target_index
    current_x, current_y =  get_window_coordinates()
    draw_point(current_x, current_y)
    try:
        if target_index <= len(path):        
            target_x, target_y = path[target_index]
            prev_x, prev_y,reached = move_to_target(current_x, current_y, target_x, target_y, prev_x, prev_y)
            if reached:
                target_index += 1
    except:
        if target_index == len(path):
            target_index = len(path)+1
            win32gui.PostMessage(hwnd, win32con.WM_KEYUP, run_key, 0)
    root.after(500, update_position)  

def save_data(x_data, y_data, filename="way.json"):
    coordinates = np.column_stack((np.array(x_data) , np.array(y_data)))
    _, unique_indices = np.unique(coordinates, axis=0, return_index=True)
    unique_coordinates = coordinates[sorted(unique_indices)]
    x_data = unique_coordinates[:, 0]
    y_data = unique_coordinates[:, 1]
    data = {"x": x_data.tolist(), "y": y_data.tolist()}
    with open(filename, "w") as f:
        json.dump(data, f)
    root.destroy()

def on_closing():
    save_data(x_data, y_data)  
root.protocol("WM_DELETE_WINDOW", on_closing)  
root.after(500, update_position)  
canvas.bind("<Button-3>", on_right_click)
root.mainloop()


