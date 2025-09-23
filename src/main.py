#!/usr/bin/env python3
"""
Grid-Based Pre-placement Tool with Blockage Support
使用網格結構和 A* 演算法的預放置工具
Modified to process close_to_target first, then pipe
Pipe stages are now treated as instance groups
Added support for target_type: sram and sram_group
"""

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
import heapq
import re
import time # 引入 time 模組用於暫停

from testcase import (
    create_test_files_with_blockages
)

# 檢測 matplotlib 版本以使用正確的 API
MATPLOTLIB_VERSION = tuple(map(int, matplotlib.__version__.split('.')[:2]))
USE_NEW_CMAP_API = MATPLOTLIB_VERSION >= (3, 7)

@dataclass
class Point:
    """座標點"""
    x: float
    y: float

    def to_grid(self, grid_size: float) -> Tuple[int, int]:
        """轉換為網格座標"""
        return (int(self.x / grid_size), int(self.y / grid_size))

    def __str__(self):
        return f"({self.x:.2f}, {self.y:.2f})"


@dataclass
class Rectangle:
    """矩形區域"""
    llx: float  # 左下 x
    lly: float  # 左下 y
    urx: float  # 右上 x
    ury: float  # 右上 y

    def contains_point(self, point: Point) -> bool:
        """檢查點是否在矩形內"""
        return (self.llx <= point.x <= self.urx and
                self.lly <= point.y <= self.ury)

    def to_grid_region(self, grid_size: float) -> Tuple[int, int, int, int]:
        """轉換為網格區域 (llx, lly, urx, ury 的網格索引)"""
        return (
            int(self.llx / grid_size),
            int(self.lly / grid_size),
            int(np.ceil(self.urx / grid_size)),
            int(np.ceil(self.ury / grid_size))
        )

    def center(self) -> Point:
        """返回矩形中心點"""
        return Point((self.llx + self.urx) / 2, (self.lly + self.ury) / 2)


@dataclass
class IPDefinition:
    """IP 定義資料結構"""
    name: str
    size_x: float
    size_y: float

    def get_size(self) -> Tuple[float, float]:
        return (self.size_x, self.size_y)

@dataclass
class Block:
    """設計中的硬區塊"""
    name: str
    boundary: Rectangle
    pin_box: Optional[Rectangle] = None

@dataclass
class Blockage:
    """阻擋區域 (現在會被視為 GRID_BLOCKED)"""
    name: str
    boundary: Rectangle

@dataclass
class IOPad:
    """I/O Pad 資料結構"""
    name: str
    boundary: Rectangle
    pin_box: Optional[Rectangle] = None

@dataclass
class MPSensor:
    """MP Sensor 資料結構"""
    name: str
    boundary: Rectangle
    pin_box: Optional[Rectangle] = None

@dataclass
class Constraint:
    """放置約束"""
    type: str  # 'close_to_target', 'pipe'
    elements: List[str] = field(default_factory=list) # 受影響的元件列表

    # close_to_target 相關
    target_type: Optional[str] = None # 'cell', 'coords', 'pin', 'sram', 'sram_group'  # Added 'sram_group'
    target_name: Optional[str] = None # 目標名稱 (cell) or space-separated list for sram_group
    target_coords: Optional[Point] = None # 目標座標 (coords)
    target_blocks: Optional[List[str]] = None  # NEW: For sram_group, list of block names

    # pipe 相關
    start: Optional[str] = None   # pipe 起點元件
    end: Optional[str] = None     # pipe 終點元件
    stages: Optional[List[List[str]]] = None  # NEW: List of stages, each stage is a list of instances

    element_type: str = 'single'  # 'single', 'instancesgroup', 'module'
    area: float = 20.0            # 僅用於 instancesgroup/module

    # 視覺化輔助
    vis_target_point: Optional[Point] = None # close_to_target 實際解析出的目標點
    vis_start_point: Optional[Point] = None  # pipe 實際解析出的起點
    vis_end_point: Optional[Point] = None    # pipe 實際解析出的終點
    vis_pipe_path: Optional[List[Tuple[int, int]]] = None  # pipe 實際使用的路徑
    vis_bounding_rect: Optional[Rectangle] = None  # NEW: For sram_group bounding rectangle

    def __str__(self):
        if self.type == 'close_to_target':
            target_info = ""
            if self.target_type == 'coords' and self.target_coords:
                target_info = f"coords={self.target_coords}"
            elif self.target_type in ['cell', 'pin', 'sram'] and self.target_name:
                target_info = f"{self.target_type}='{self.target_name}'"
            elif self.target_type == 'sram_group' and self.target_blocks:  # NEW
                target_info = f"sram_group={self.target_blocks}"
            return f"Constraint(Type='{self.type}', Target={target_info})"
        elif self.type == 'pipe':
            num_stages = len(self.stages) if self.stages else 0
            return f"Constraint(Type='{self.type}', Start='{self.start}', End='{self.end}', Stages={num_stages})"
        else:
            return super().__str__() # 對於未知類型，使用預設的 dataclass 字串表示


class GridBasedPlacer:
    """基於網格的放置器"""

    def __init__(self, grid_size: float = 50.0, root_module_name: str = "uDue1/u_socss_0"):
        """
        初始化
        Args:
            grid_size: 網格大小（微米）
            root_module_name: TVC JSON 中根模組的名稱
        """
        self.grid_size = grid_size
        self.root_module_name = root_module_name

        # 設計資訊
        self.die_boundary: Optional[Rectangle] = None
        self.core_area: Optional[Rectangle] = None
        self.blocks: Dict[str, Block] = {}
        self.blockages: Dict[str, Blockage] = {} # 僅用於載入和視覺化，網格中會轉為 GRID_BLOCKED
        self.io_pads: Dict[str, IOPad] = {}
        self.mp_sensors: Dict[str, MPSensor] = {}
        self.ip_definitions: Dict[str, IPDefinition] = {}

        # 網格地圖
        self.grid_map: Optional[np.ndarray] = None
        self.grid_width: int = 0
        self.grid_height: int = 0

        # 約束和放置結果
        self.constraints: List[Constraint] = []
        self.placements: Dict[str, any] = {} # 儲存實際放置的元件或群組資訊
        self.instance_locations: Dict[str, Point] = {}
        self.failed_constraints: List[str] = []

        # 網格狀態定義
        self.GRID_FREE = 0      # 可用
        self.GRID_BLOCKED = 1   # 被阻擋 (硬區塊、blockage、Core 外、已放置元件、保留區域)
        self.GRID_RESERVED = 2  # 臨時保留 (用於群組分配)

        # 視覺化相關成員變數
        self.fig = None
        self.ax_grid = None
        self.ax_placement = None
        self.grid_img = None # 用於更新網格圖
        self.placement_artists = [] # 儲存放置結果的 matplotlib 物件，以便清除
        self.current_constraint_text = None # 顯示當前處理的約束
        self.cmap_constraints = None # 約束顏色映射

    def is_valid_grid_coord(self, x: int, y: int) -> bool:
        """統一的網格座標有效性檢查"""
        return 0 <= x < self.grid_width and 0 <= y < self.grid_height

    def load_tvc_json(self, filepath: str):
        """載入 TVC JSON 檔案"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        if 'IPS' in data:
            for ip_name, ip_data in data['IPS'].items():
                if 'SIZE_X' in ip_data and 'SIZE_Y' in ip_data:
                    self.ip_definitions[ip_name] = IPDefinition(
                        name=ip_name,
                        size_x=float(ip_data['SIZE_X']),
                        size_y=float(ip_data['SIZE_Y'])
                    )
            print(f"    ✓ Loaded {len(self.ip_definitions)} IP definitions.")

        tvc_root = data['S1_output'][self.root_module_name]
        top_shape = tvc_root['TOPShape']

        # 晶片邊界
        die_coords = top_shape['DieCoords']
        self.die_boundary = Rectangle(
            die_coords[0][0], die_coords[0][1],
            die_coords[2][0], die_coords[2][1]
        )

        # 核心區域
        core_coords = top_shape['CoreCoords']
        self.core_area = Rectangle(
            core_coords[0][0], core_coords[0][1],
            core_coords[2][0], core_coords[2][1]
        )

        # 區塊
        if 'BlockShapes' in tvc_root:
            for block_data in tvc_root['BlockShapes']:
                coords = block_data['Coords']
                block = Block(
                    name=block_data['BlockName'],
                    boundary=Rectangle(
                        min(c[0] for c in coords),
                        min(c[1] for c in coords),
                        max(c[0] for c in coords),
                        max(c[1] for c in coords)
                    )
                )

                # Pin box 處理
                if 'PinCoords' in block_data and block_data['PinCoords']:
                    pin_list = block_data['PinCoords']
                    if pin_list and pin_list[0]: # 檢查列表和第一個元素是否非空
                        first_pin_coords = pin_list[0]
                        block.pin_box = Rectangle(
                            min(c[0] for c in first_pin_coords),
                            min(c[1] for c in first_pin_coords),
                            max(c[0] for c in first_pin_coords),
                            max(c[1] for c in first_pin_coords)
                        )
                self.blocks[block.name] = block

        if 'Solution_MP_SENSOR' in data and self.root_module_name in data['Solution_MP_SENSOR']:
            if 'MP_SENSOR' in data['Solution_MP_SENSOR'][self.root_module_name]:
                mp_sensors = data['Solution_MP_SENSOR'][self.root_module_name]['MP_SENSOR']
                for sensor_data in mp_sensors:
                    name = sensor_data['Name']
                    cell_name = sensor_data['CellName']
                    location = sensor_data['Location'] # [llx, lly]

                    if cell_name in self.ip_definitions:
                        size_x, size_y = self.ip_definitions[cell_name].get_size()
                        llx, lly = location[0], location[1]
                        urx, ury = llx + size_x, lly + size_y

                        self.mp_sensors[name] = MPSensor(
                            name=name,
                            boundary=Rectangle(llx, lly, urx, ury),
                            pin_box=None  # 初始為None，之後可實作
                        )
                    else:
                        print(f"警告: MP_SENSOR '{name}' 的 CellName '{cell_name}' 在 IP 尺寸中未找到，跳過。")
            else:
                print(f"警告: '{self.root_module_name}' 下沒有 'MP_SENSOR' 資訊。")
        else:
            print(f"警告: 'Solution_MP_SENSOR' 或 '{self.root_module_name}' 在其中未找到。")

        # 載入 Blockages
        if 'Blockages' in top_shape:
            blockages_data = top_shape['Blockages']
            for idx, blockage_coords_list in enumerate(blockages_data):
                coords = blockage_coords_list
                if coords and coords[-1] == coords[0]:
                    coords = coords[:-1] # 移除重複的最後一個點

                if not coords: continue # 空的 blockage 列表

                x_coords = [c[0] for c in coords]
                y_coords = [c[1] for c in coords]

                blockage = Blockage(
                    name=f"BLOCKAGE_{idx}",
                    boundary=Rectangle(
                        min(x_coords),
                        min(y_coords),
                        max(x_coords),
                        max(y_coords)
                    )
                )
                self.blockages[blockage.name] = blockage

        if 'Solution_GPIO' in data and self.root_module_name in data['Solution_GPIO']:
            if 'GPIO' in data['Solution_GPIO'][self.root_module_name]:
                gpio_pads = data['Solution_GPIO'][self.root_module_name]['GPIO']
                for pad_data in gpio_pads:
                    name = pad_data['Name']
                    cell_name = pad_data['CellName']
                    location = pad_data['Location'] # [llx, lly]

                    if cell_name in self.ip_definitions:
                        size_x, size_y = self.ip_definitions[cell_name].get_size()
                        llx, lly = location[0], location[1]
                        urx, ury = llx + size_x, lly + size_y

                        self.io_pads[name] = IOPad(
                            name=name,
                            boundary=Rectangle(llx, lly, urx, ury),
                            pin_box=None  # 初始為None，之後可實作
                        )
                    else:
                        print(f"警告: I/O Pad '{name}' 的 CellName '{cell_name}' 在 IP 尺寸中未找到，跳過。")
            else:
                print(f"警告: '{self.root_module_name}' 下沒有 'GPIO' 資訊。")
        else:
            print(f"警告: 'Solution_GPIO' 或 '{self.root_module_name}' 在其中未找到。")

    def load_constraints(self, filepath: str):
        """載入約束檔案"""
        with open(filepath, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line.startswith('Preplace Type:'):
                constraint = Constraint(type='unknown')

                if 'close to target' in line:
                    constraint.type = 'close_to_target'
                elif 'pipe' in line:
                    constraint.type = 'pipe'
                    constraint.element_type = 'instancesgroup'  # CHANGED: Default for pipe is now instancesgroup
                    constraint.stages = []  # NEW: Initialize stages list

                i += 1
                while i < len(lines):
                    line = lines[i].strip()
                    if not line: # 空行結束一個約束
                        break
                    if line.startswith('Preplace Type:'): # 遇到下一個約束
                        i -= 1 # 回退一行，讓外層迴圈處理
                        break

                    # close_to_target 相關解析
                    if constraint.type == 'close_to_target':
                        if line.startswith('Target Type:'):
                            constraint.target_type = line.split(':')[1].strip()
                        elif line.startswith('Target:'):
                            target_value = line.split(':')[1].strip()
                            if constraint.target_type == 'coords':
                                match_coords = re.search(r'\[(\S+),\s*(\S+)\]', target_value)
                                if match_coords:
                                    constraint.target_coords = Point(
                                        float(match_coords.group(1)),
                                        float(match_coords.group(2))
                                    )
                            elif constraint.target_type == 'sram_group':  # NEW: Parse multiple blocks
                                constraint.target_blocks = target_value.split()
                                constraint.target_name = target_value  # Keep original string for compatibility
                            else: # 'cell' or 'pin' or 'sram'
                                constraint.target_name = target_value
                        elif line.startswith('Element Type:'):
                            constraint.element_type = line.split(':')[1].strip()
                        elif line.startswith('Area:'):
                            constraint.area = float(line.split(':')[1].strip())
                        elif line.startswith('Element:'):
                            constraint.elements = line.split(':')[1].strip().split()

                    # pipe 相關解析 - MODIFIED
                    elif constraint.type == 'pipe':
                        if line.startswith('Start:'):
                            constraint.start = line.split(':')[1].strip()
                        elif line.startswith('End:'):
                            constraint.end = line.split(':')[1].strip()
                        elif line.startswith('Stage'): # Stage1, Stage2, ...
                            stage_elements = line.split(':')[1].strip().split()
                            if stage_elements:  # Only add non-empty stages
                                constraint.stages.append(stage_elements)  # NEW: Add as a stage
                                constraint.elements.extend(stage_elements) # Keep for compatibility
                        # No longer process Element Type or Area for pipe

                    i += 1

                # Only add constraint if it has elements (or stages for pipe)
                if constraint.type == 'pipe' and constraint.stages:
                    self.constraints.append(constraint)
                elif constraint.type == 'close_to_target' and constraint.elements:
                    self.constraints.append(constraint)

            i += 1

    def build_grid_map(self):
        """建立網格地圖"""
        if not self.die_boundary or not self.core_area:
            raise ValueError("Die boundary or core area not loaded.")

        # 計算網格大小
        self.grid_width = int(np.ceil(self.die_boundary.urx / self.grid_size))
        self.grid_height = int(np.ceil(self.die_boundary.ury / self.grid_size))

        # 初始化網格（0 = 可用 GRID_FREE）
        self.grid_map = np.zeros((self.grid_height, self.grid_width), dtype=int)

        # 標記核心區域外為不可用 (GRID_BLOCKED)
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                px = (x + 0.5) * self.grid_size
                py = (y + 0.5) * self.grid_size

                if not self.core_area.contains_point(Point(px, py)):
                    self.grid_map[y, x] = self.GRID_BLOCKED

        # 標記 blockage 佔用的網格 (GRID_BLOCKED) - 先於 Blocks 標記
        for blockage in self.blockages.values():
            grid_region = blockage.boundary.to_grid_region(self.grid_size)
            x1, y1, x2, y2 = grid_region

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(self.grid_width, x2)
            y2 = min(self.grid_height, y2)

            for gy in range(y1, y2):
                for gx in range(x1, x2):
                    self.grid_map[gy, gx] = self.GRID_BLOCKED

        # 標記區塊佔用的網格 (GRID_BLOCKED)
        for block in self.blocks.values():
            grid_region = block.boundary.to_grid_region(self.grid_size)
            x1, y1, x2, y2 = grid_region

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(self.grid_width, x2)
            y2 = min(self.grid_height, y2)

            for gy in range(y1, y2):
                for gx in range(x1, x2):
                    self.grid_map[gy, gx] = self.GRID_BLOCKED

        for sensor in self.mp_sensors.values():
            grid_region = sensor.boundary.to_grid_region(self.grid_size)
            x1, y1, x2, y2 = grid_region

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(self.grid_width, x2)
            y2 = min(self.grid_height, y2)

            for gy in range(y1, y2):
                for gx in range(x1, x2):
                    self.grid_map[gy, gx] = self.GRID_BLOCKED

    def find_max_rectangle_in_region(self, bounding_rect: Rectangle) -> Optional[Rectangle]:
        """
        Find the largest rectangle of FREE grids within the given bounding rectangle.
        Uses an approximate algorithm for efficiency.
        """
        # Convert bounding rectangle to grid coordinates
        grid_llx, grid_lly, grid_urx, grid_ury = bounding_rect.to_grid_region(self.grid_size)

        # Clamp to valid grid bounds
        grid_llx = max(0, grid_llx)
        grid_lly = max(0, grid_lly)
        grid_urx = min(self.grid_width, grid_urx)
        grid_ury = min(self.grid_height, grid_ury)

        if grid_llx >= grid_urx or grid_lly >= grid_ury:
            return None

        # Extract the sub-grid within the bounding rectangle
        sub_grid = self.grid_map[grid_lly:grid_ury, grid_llx:grid_urx]
        rows, cols = sub_grid.shape

        if rows == 0 or cols == 0:
            return None

        # Use a simplified maximal rectangle algorithm
        # For each row, calculate the height of consecutive FREE cells above
        heights = np.zeros((rows, cols), dtype=int)

        for i in range(rows):
            for j in range(cols):
                if sub_grid[i, j] == self.GRID_FREE:
                    if i == 0:
                        heights[i, j] = 1
                    else:
                        heights[i, j] = heights[i-1, j] + 1
                else:
                    heights[i, j] = 0

        max_area = 0
        best_rect = None
        best_distance = float('inf')
        bounding_center = bounding_rect.center()

        # For each row, find the maximum rectangle using histogram approach
        for i in range(rows):
            # Find maximum rectangle in histogram heights[i]
            for j in range(cols):
                if heights[i, j] == 0:
                    continue

                # Expand left and right to find width
                width = 1
                min_height = heights[i, j]

                # Expand left
                left = j
                while left > 0 and heights[i, left-1] > 0:
                    left -= 1
                    min_height = min(min_height, heights[i, left])

                # Expand right
                right = j
                while right < cols - 1 and heights[i, right+1] > 0:
                    right += 1
                    min_height = min(min_height, heights[i, right])

                # Calculate area
                width = right - left + 1
                area = width * min_height

                # Calculate the actual rectangle coordinates
                rect_grid_llx = grid_llx + left
                rect_grid_lly = grid_lly + i - min_height + 1
                rect_grid_urx = grid_llx + right + 1
                rect_grid_ury = grid_lly + i + 1

                # Convert back to real coordinates
                rect = Rectangle(
                    rect_grid_llx * self.grid_size,
                    rect_grid_lly * self.grid_size,
                    rect_grid_urx * self.grid_size,
                    rect_grid_ury * self.grid_size
                )

                # Calculate distance to bounding center
                rect_center = rect.center()
                distance = ((rect_center.x - bounding_center.x)**2 +
                           (rect_center.y - bounding_center.y)**2)**0.5

                # Choose the rectangle with largest area, or closest to center if tied
                if area > max_area or (area == max_area and distance < best_distance):
                    max_area = area
                    best_rect = rect
                    best_distance = distance

        return best_rect

    def find_nearest_free_grid(self, target: Point) -> Optional[Tuple[int, int]]:
        """
        找到最近的可用網格 (GRID_FREE)。
        使用 BFS 進行搜尋。
        """
        target_grid_x, target_grid_y = target.to_grid(self.grid_size)

        # 調整目標網格到有效範圍內
        target_grid_x = max(0, min(target_grid_x, self.grid_width - 1))
        target_grid_y = max(0, min(target_grid_y, self.grid_height - 1))

        # 如果目標點本身可用
        if self.grid_map[target_grid_y, target_grid_x] == self.GRID_FREE:
            return (target_grid_x, target_grid_y)

        # BFS 搜尋最近的可用網格
        visited = set()
        queue = [(target_grid_x, target_grid_y)]

        while queue:
            x, y = queue.pop(0)

            if (x, y) in visited:
                continue
            visited.add((x, y))

            if self.grid_map[y, x] == self.GRID_FREE:
                return (x, y)

            # 探索四個方向 (從中心向外擴展)
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if self.is_valid_grid_coord(nx, ny) and (nx, ny) not in visited:
                    queue.append((nx, ny))

        return None # 找不到可用網格

    def a_star_path(self, start: Tuple[int, int],
                    end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        使用 A* 演算法尋找路徑，只避開 GRID_BLOCKED。
        允許通過 GRID_FREE 和 GRID_RESERVED 區域。
        Returns:
            網格座標列表，如果找不到路徑則返回 None。
        """
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = []
        heapq.heappush(open_set, (0 + heuristic(start, end), 0, start, [start]))

        g_scores = {start: 0}
        visited_nodes = set()

        while open_set:
            f_score, g_score, current, path = heapq.heappop(open_set)

            if current in visited_nodes:
                continue
            visited_nodes.add(current)

            if current == end:
                return path

            # 探索鄰居
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                nx, ny = neighbor

                if (self.is_valid_grid_coord(nx, ny) and
                    self.grid_map[ny, nx] != self.GRID_BLOCKED and # 可以走 FREE 或 RESERVED
                    neighbor not in visited_nodes):

                    tentative_g_score = g_score + 1

                    if neighbor not in g_scores or tentative_g_score < g_scores[neighbor]:
                        g_scores[neighbor] = tentative_g_score
                        h_score = heuristic(neighbor, end)
                        f_score = tentative_g_score + h_score
                        heapq.heappush(open_set, (f_score, tentative_g_score, neighbor, path + [neighbor]))

        return None # 找不到路徑

    def get_target_point_for_constraint(self, constraint: Constraint) -> Optional[Point]:
        """根據約束解析出目標點的實際座標"""
        if constraint.target_type == 'coords' and constraint.target_coords:
            return constraint.target_coords
        elif constraint.target_type == 'pin':
            path_parts = constraint.target_name.split('/')
            if len(path_parts) > 1:
                potential_block_path = '/'.join(path_parts[:-1])
                if potential_block_path in self.blocks:
                    block = self.blocks[potential_block_path]
                    if block.pin_box:
                        return block.pin_box.center()
                    else:
                        return block.boundary.center()
            print(f"警告: Pin '{constraint.target_name}' 無法解析出對應的 block。")
            print(f"      可用的 blocks: {list(self.blocks.keys())}")
            print(f"      可用的 I/O pads: {list(self.io_pads.keys())}")
            print(f"      可用的 MP sensors: {list(self.mp_sensors.keys())}")
        elif constraint.target_type == 'cell':
            # 對於 cell 類型，檢查pin_box是否存在
            if constraint.target_name in self.blocks:
                block = self.blocks[constraint.target_name]
                # 優先使用pin_box，如果不存在則使用boundary中心
                if block.pin_box:
                    return block.pin_box.center()
                else:
                    return block.boundary.center()
            elif constraint.target_name in self.io_pads:
                io_pad = self.io_pads[constraint.target_name]
                # 優先使用pin_box，如果不存在則使用boundary中心
                if io_pad.pin_box:
                    return io_pad.pin_box.center()
                else:
                    return io_pad.boundary.center()
            elif constraint.target_name in self.mp_sensors:
                mp_sensor = self.mp_sensors[constraint.target_name]
                # 優先使用pin_box，如果不存在則使用boundary中心
                if mp_sensor.pin_box:
                    return mp_sensor.pin_box.center()
                else:
                    return mp_sensor.boundary.center()
        elif constraint.target_type == 'sram':  # Existing SRAM target type
            # For SRAM, we don't return a single target point
            # The processing is handled differently in process_constraints
            return None
        elif constraint.target_type == 'sram_group':  # NEW: sram_group doesn't have a single target point
            return None
        return None

    def determine_pin_edge(self, block: Block) -> str:
        """
        Determine which edge of the block contains the pin_box.
        Returns: 'left', 'right', 'top', or 'bottom'
        """
        if not block.pin_box:
            return 'bottom'  # Default if no pin_box

        pin_center = block.pin_box.center()
        block_center = block.boundary.center()

        # Calculate distances from pin center to each edge
        dist_left = abs(pin_center.x - block.boundary.llx)
        dist_right = abs(pin_center.x - block.boundary.urx)
        dist_bottom = abs(pin_center.y - block.boundary.lly)
        dist_top = abs(pin_center.y - block.boundary.ury)

        # Find the minimum distance
        min_dist = min(dist_left, dist_right, dist_bottom, dist_top)

        # Return the corresponding edge
        if min_dist == dist_left:
            return 'left'
        elif min_dist == dist_right:
            return 'right'
        elif min_dist == dist_bottom:
            return 'bottom'
        else:
            return 'top'

    def allocate_sram_region(self, target_block: Block, area: float) -> Optional[Rectangle]:
        """
        Allocate a rectangular region for SRAM placement.
        The region will have one edge matching the pin edge of the target block.
        """
        pin_edge = self.determine_pin_edge(target_block)

        # Calculate dimensions based on pin edge
        if pin_edge in ['left', 'right']:
            # Vertical edge - height matches block height
            edge_length = target_block.boundary.ury - target_block.boundary.lly
            other_length = area / edge_length

            if pin_edge == 'left':
                # Place to the left of the block
                return Rectangle(
                    target_block.boundary.llx - other_length,
                    target_block.boundary.lly,
                    target_block.boundary.llx,
                    target_block.boundary.ury
                )
            else:  # right
                # Place to the right of the block
                return Rectangle(
                    target_block.boundary.urx,
                    target_block.boundary.lly,
                    target_block.boundary.urx + other_length,
                    target_block.boundary.ury
                )
        else:  # top or bottom
            # Horizontal edge - width matches block width
            edge_length = target_block.boundary.urx - target_block.boundary.llx
            other_length = area / edge_length

            if pin_edge == 'bottom':
                # Place below the block
                return Rectangle(
                    target_block.boundary.llx,
                    target_block.boundary.lly - other_length,
                    target_block.boundary.urx,
                    target_block.boundary.lly
                )
            else:  # top
                # Place above the block
                return Rectangle(
                    target_block.boundary.llx,
                    target_block.boundary.ury,
                    target_block.boundary.urx,
                    target_block.boundary.ury + other_length
                )

    def allocate_region(self, area: float, near_point: Point) -> Optional[Rectangle]:
        """為實例群組分配矩形區域"""
        grids_needed = int(np.ceil(area / (self.grid_size * self.grid_size)))
        side_grids_min = int(np.ceil(np.sqrt(grids_needed)))

        target_grid_x, target_grid_y = near_point.to_grid(self.grid_size)

        best_region_rect = None
        min_dist_sq = float('inf')

        # 限制搜尋範圍，避免無限擴展
        search_limit = max(self.grid_width, self.grid_height) // 2

        # 從目標點開始，螺旋向外搜尋
        for radius in range(search_limit + 1):
            # 遍歷以 (target_grid_x, target_grid_y) 為中心，半徑為 radius 的方形邊界
            # 確保從中心點開始向外擴散， radius=0 處理中心點，radius=1 處理周圍8個點
            current_points_to_check = set()
            if radius == 0:
                current_points_to_check.add((target_grid_x, target_grid_y))
            else:
                for dx in range(-radius, radius + 1):
                    # 檢查邊界點
                    if 0 <= target_grid_y - radius < self.grid_height: # 底部邊
                        current_points_to_check.add((target_grid_x + dx, target_grid_y - radius))
                    if 0 <= target_grid_y + radius < self.grid_height: # 頂部邊
                        current_points_to_check.add((target_grid_x + dx, target_grid_y + radius))
                for dy in range(-radius + 1, radius): # 左右邊，避免重複角點
                    if 0 <= target_grid_x - radius < self.grid_width:
                        current_points_to_check.add((target_grid_x - radius, target_grid_y + dy))
                    if 0 <= target_grid_x + radius < self.grid_width:
                        current_points_to_check.add((target_grid_x + radius, target_grid_y + dy))

            for ll_grid_x, ll_grid_y in current_points_to_check:
                current_side_grids = side_grids_min
                ur_grid_x = ll_grid_x + current_side_grids
                ur_grid_y = ll_grid_y + current_side_grids

                if not (0 <= ll_grid_x < ur_grid_x <= self.grid_width and
                        0 <= ll_grid_y < ur_grid_y <= self.grid_height):
                    continue

                is_free = True
                for y_grid in range(ll_grid_y, ur_grid_y):
                    for x_grid in range(ll_grid_x, ur_grid_x):
                        if not self.is_valid_grid_coord(x_grid, y_grid):
                            is_free = False # 超出網格範圍
                            break
                        if self.grid_map[y_grid, x_grid] != self.GRID_FREE:
                            is_free = False
                            break
                    if not is_free:
                        break

                if is_free:
                    # 計算此區域中心到 near_point 的距離
                    region_center_x = (ll_grid_x + current_side_grids / 2.0) * self.grid_size
                    region_center_y = (ll_grid_y + current_side_grids / 2.0) * self.grid_size

                    dist_sq = (region_center_x - near_point.x)**2 + \
                              (region_center_y - near_point.y)**2

                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        best_region_rect = Rectangle(
                            ll_grid_x * self.grid_size,
                            ll_grid_y * self.grid_size,
                            ur_grid_x * self.grid_size,
                            ur_grid_y * self.grid_size
                        )

            if best_region_rect: # 如果當前 radius 找到了至少一個 region，就不再擴大 radius 搜尋
                break # 確保找到的是最靠近的，而不是最遠的

        if best_region_rect:
            # FIXED: 使用正確的 to_grid_region 方法，返回 4 個值
            llx_grid, lly_grid, urx_grid, ury_grid = best_region_rect.to_grid_region(self.grid_size)

            # 標記區域為已保留
            for y_grid in range(lly_grid, ury_grid):
                for x_grid in range(llx_grid, urx_grid):
                    if self.is_valid_grid_coord(x_grid, y_grid):
                        self.grid_map[y_grid, x_grid] = self.GRID_RESERVED

        return best_region_rect

    def allocate_single_grid_region(self, near_point: Point) -> Optional[Rectangle]:
        """Allocate a single grid cell region near the given point"""
        grid_x, grid_y = near_point.to_grid(self.grid_size)

        # Adjust to valid range
        grid_x = max(0, min(grid_x, self.grid_width - 1))
        grid_y = max(0, min(grid_y, self.grid_height - 1))

        # If target grid is free, use it
        if self.grid_map[grid_y, grid_x] == self.GRID_FREE:
            self.grid_map[grid_y, grid_x] = self.GRID_RESERVED
            return Rectangle(
                grid_x * self.grid_size,
                grid_y * self.grid_size,
                (grid_x + 1) * self.grid_size,
                (grid_y + 1) * self.grid_size
            )

        # Otherwise, find nearest free grid
        nearest = self.find_nearest_free_grid(near_point)
        if nearest:
            gx, gy = nearest
            self.grid_map[gy, gx] = self.GRID_RESERVED
            return Rectangle(
                gx * self.grid_size,
                gy * self.grid_size,
                (gx + 1) * self.grid_size,
                (gy + 1) * self.grid_size
            )

        return None

    def get_instance_position(self, inst_name: str) -> Optional[Point]:
        """
        Get the position of an instance.
        First check instance_locations (for individual instances and group members),
        then check blocks, IO pads, and MP sensors.
        """
        # Check in instance_locations first (includes both single instances and group members)
        if inst_name in self.instance_locations:
            return self.instance_locations[inst_name]

        # Check in blocks
        if inst_name in self.blocks:
            return self.blocks[inst_name].boundary.center()

        # Check in IO pads
        if inst_name in self.io_pads:
            return self.io_pads[inst_name].boundary.center()

        # FIX 4: Check in MP sensors
        if inst_name in self.mp_sensors:
            return self.mp_sensors[inst_name].boundary.center()

        return None

    def process_constraints(self):
        """
        Process constraints in order: first all close_to_target, then all pipe.
        Update visualization after each constraint.
        """
        group_id = 0

        # Sort constraints: close_to_target first, then pipe
        sorted_constraints = sorted(self.constraints,
                                   key=lambda c: 0 if c.type == 'close_to_target' else 1)
        self.constraints = sorted_constraints

        # Initialize visualization (if not already initialized)
        if self.fig is None:
            self._setup_placement_visualization()

        for idx, constraint in enumerate(self.constraints):
            print(f"處理約束 C{idx}: {constraint}")

            # Update current constraint text label
            if self.current_constraint_text:
                self.current_constraint_text.remove()
            self.current_constraint_text = self.ax_placement.text(
                0.5, 0.98, f"Processing C{idx}: {constraint.type}",
                transform=self.ax_placement.transAxes, va='top', ha='center',
                fontsize=12, color='red',
                bbox=dict(facecolor='yellow', alpha=0.9, edgecolor='red', boxstyle='round,pad=0.3')
            )

            try:
                if constraint.type == 'close_to_target':
                    # Handle SRAM_GROUP target type - NEW
                    if constraint.target_type == 'sram_group':
                        if not constraint.target_blocks:
                            self.failed_constraints.append(
                                f"Constraint C{idx} for elements {constraint.elements}: "
                                f"No target blocks specified for sram_group."
                            )
                            self._update_placement_visualization(idx)
                            continue

                        # Check all blocks exist
                        missing_blocks = []
                        for block_name in constraint.target_blocks:
                            if block_name not in self.blocks:
                                missing_blocks.append(block_name)

                        if missing_blocks:
                            self.failed_constraints.append(
                                f"Constraint C{idx} for elements {constraint.elements}: "
                                f"Target blocks not found: {missing_blocks}"
                            )
                            self._update_placement_visualization(idx)
                            continue

                        # Calculate bounding rectangle for all target blocks
                        min_llx = float('inf')
                        min_lly = float('inf')
                        max_urx = float('-inf')
                        max_ury = float('-inf')

                        for block_name in constraint.target_blocks:
                            block = self.blocks[block_name]
                            min_llx = min(min_llx, block.boundary.llx)
                            min_lly = min(min_lly, block.boundary.lly)
                            max_urx = max(max_urx, block.boundary.urx)
                            max_ury = max(max_ury, block.boundary.ury)

                        bounding_rect = Rectangle(min_llx, min_lly, max_urx, max_ury)
                        constraint.vis_bounding_rect = bounding_rect  # Save for visualization

                        # Find largest free rectangle within bounding rect
                        region = self.find_max_rectangle_in_region(bounding_rect)

                        if region:
                            actual_area = (region.urx - region.llx) * (region.ury - region.lly)
                            required_area = constraint.area

                            # Check if area is sufficient and show warning if not
                            if actual_area < required_area:
                                print(f"  ⚠ Warning: Found area ({actual_area:.2f}) is smaller than required ({required_area:.2f})")

                            group_name = f"TVC_SRAM_MULTI_GROUP_{group_id}"
                            group_id += 1

                            # Mark the region as RESERVED in the grid
                            llx_grid, lly_grid, urx_grid, ury_grid = region.to_grid_region(self.grid_size)
                            for y_grid in range(lly_grid, ury_grid):
                                for x_grid in range(llx_grid, urx_grid):
                                    if self.is_valid_grid_coord(x_grid, y_grid):
                                        if self.grid_map[y_grid, x_grid] == self.GRID_FREE:
                                            self.grid_map[y_grid, x_grid] = self.GRID_RESERVED

                            self.placements[group_name] = {
                                'type': 'group',
                                'region': region,
                                'instances': constraint.elements,
                                'constraint_idx': idx,
                                'is_sram_group': True  # Mark as SRAM multi-block group
                            }

                            # Record each instance in the group at the center of the region
                            center_point = region.center()
                            for element in constraint.elements:
                                self.instance_locations[element] = center_point
                                print(f"  → Placed {element} in SRAM multi-block group {group_name} at center {center_point}")

                            # For visualization, set target point at bounding rect center
                            constraint.vis_target_point = bounding_rect.center()
                        else:
                            self.failed_constraints.append(
                                f"Constraint C{idx} for elements {constraint.elements}: "
                                f"Cannot find any free rectangle within bounding region of blocks {constraint.target_blocks}."
                            )
                    # Handle SRAM target type specially
                    elif constraint.target_type == 'sram':
                        if constraint.target_name not in self.blocks:
                            self.failed_constraints.append(
                                f"Constraint C{idx} for elements {constraint.elements}: "
                                f"Target block '{constraint.target_name}' not found for SRAM placement."
                            )
                            self._update_placement_visualization(idx)
                            continue

                        target_block = self.blocks[constraint.target_name]
                        group_name = f"TVC_SRAM_GROUP_{group_id}"
                        group_id += 1

                        # Allocate SRAM region adjacent to pin edge
                        region = self.allocate_sram_region(target_block, constraint.area)

                        if region:
                            # Mark the region as RESERVED in the grid
                            llx_grid, lly_grid, urx_grid, ury_grid = region.to_grid_region(self.grid_size)
                            for y_grid in range(lly_grid, ury_grid):
                                for x_grid in range(llx_grid, urx_grid):
                                    if self.is_valid_grid_coord(x_grid, y_grid):
                                        self.grid_map[y_grid, x_grid] = self.GRID_RESERVED

                            self.placements[group_name] = {
                                'type': 'group',
                                'region': region,
                                'instances': constraint.elements,
                                'constraint_idx': idx,
                                'is_sram': True  # Mark as SRAM group for visualization
                            }

                            # Record each instance in the group at the center of the region
                            center_point = region.center()
                            for element in constraint.elements:
                                self.instance_locations[element] = center_point
                                print(f"  → Placed {element} in SRAM group {group_name} at center {center_point}")

                            # For visualization, set a target point at the pin center
                            if target_block.pin_box:
                                constraint.vis_target_point = target_block.pin_box.center()
                            else:
                                constraint.vis_target_point = target_block.boundary.center()
                        else:
                            self.failed_constraints.append(
                                f"Constraint C{idx} ({group_name}, elements: {constraint.elements}): "
                                f"Cannot allocate SRAM region for area {constraint.area}."
                            )
                    else:  # Original close_to_target logic
                        target_point = self.get_target_point_for_constraint(constraint)

                        if not target_point:
                            self.failed_constraints.append(
                                f"Constraint C{idx} for elements {constraint.elements}: "
                                f"Target '{constraint.target_name or str(constraint.target_coords)}' not found or invalid."
                            )
                            self._update_placement_visualization(idx)
                            continue

                        constraint.vis_target_point = target_point

                        # 處理 module 和 instancesgroup 相同
                        if constraint.element_type in ['instancesgroup', 'module']:
                            group_name = f"TVC_INST_GROUP_{group_id}"
                            group_id += 1

                            region = self.allocate_region(constraint.area, target_point)

                            if region:
                                self.placements[group_name] = {
                                    'type': 'group',
                                    'region': region,
                                    'instances': constraint.elements,
                                    'constraint_idx': idx
                                }

                                # NEW: Record each instance in the group at the center of the region
                                center_point = region.center()
                                for element in constraint.elements:
                                    self.instance_locations[element] = center_point
                                    print(f"  → Placed {element} in group {group_name} at center {center_point}")
                            else:
                                self.failed_constraints.append(
                                    f"Constraint C{idx} ({group_name}, elements: {constraint.elements}): "
                                    f"Cannot allocate region for area {constraint.area} (no free space found)."
                                )

                elif constraint.type == 'pipe':
                    # Get start and end positions from previously placed instances
                    start_pos = self.get_instance_position(constraint.start)
                    end_pos = self.get_instance_position(constraint.end)

                    # If start or end not found, use fallback positions
                    if not start_pos:
                        start_grid_candidate = self.find_nearest_free_grid(Point(
                            self.core_area.llx + self.grid_size,
                            self.core_area.lly + self.grid_size
                        ))
                        if start_grid_candidate:
                            start_pos = Point(
                                (start_grid_candidate[0] + 0.5) * self.grid_size,
                                (start_grid_candidate[1] + 0.5) * self.grid_size
                            )
                        else:
                            start_pos = Point(
                                self.core_area.llx + self.grid_size,
                                self.core_area.lly + self.grid_size
                            )
                        print(f"警告: Constraint C{idx}: Start element '{constraint.start}' not found or placed, "
                              f"using a point near ({start_pos.x:.2f}, {start_pos.y:.2f}).")

                    if not end_pos:
                        end_grid_candidate = self.find_nearest_free_grid(Point(
                            self.core_area.urx - self.grid_size,
                            self.core_area.ury - self.grid_size
                        ))
                        if end_grid_candidate:
                            end_pos = Point(
                                (end_grid_candidate[0] + 0.5) * self.grid_size,
                                (end_grid_candidate[1] + 0.5) * self.grid_size
                            )
                        else:
                            end_pos = Point(
                                self.core_area.urx - self.grid_size,
                                self.core_area.ury - self.grid_size
                            )
                        print(f"警告: Constraint C{idx}: End element '{constraint.end}' not found or placed, "
                              f"using a point near ({end_pos.x:.2f}, {end_pos.y:.2f}).")

                    constraint.vis_start_point = start_pos
                    constraint.vis_end_point = end_pos

                    # Directly convert start/end positions to grid coordinates
                    start_grid = start_pos.to_grid(self.grid_size)
                    end_grid = end_pos.to_grid(self.grid_size)

                    # Ensure grid coordinates are within bounds
                    start_grid = (
                        max(0, min(start_grid[0], self.grid_width - 1)),
                        max(0, min(start_grid[1], self.grid_height - 1))
                    )
                    end_grid = (
                        max(0, min(end_grid[0], self.grid_width - 1)),
                        max(0, min(end_grid[1], self.grid_height - 1))
                    )

                    path = self.a_star_path(start_grid, end_grid)

                    if path:
                        # Save the actual path for visualization
                        constraint.vis_pipe_path = path

                        # Process pipe stages as groups
                        num_stages = len(constraint.stages) if constraint.stages else 0
                        if num_stages == 0:
                            self._update_placement_visualization(idx)
                            continue

                        # Distribute stage groups along the path
                        path_step = max(1, len(path) // (num_stages + 1))

                        for stage_idx, stage_instances in enumerate(constraint.stages):
                            # Determine position for this stage group
                            idx_on_path = min((stage_idx + 1) * path_step, len(path) - 1)
                            grid_point = path[idx_on_path]

                            # Check if this grid point is FREE
                            if self.grid_map[grid_point[1], grid_point[0]] != self.GRID_FREE:
                                # Find nearest free grid if the path point is not free
                                stage_point = Point(
                                    (grid_point[0] + 0.5) * self.grid_size,
                                    (grid_point[1] + 0.5) * self.grid_size
                                )
                                nearest_free = self.find_nearest_free_grid(stage_point)

                                if nearest_free:
                                    grid_point = nearest_free
                                    print(f"  → Stage {stage_idx} moved to nearest free grid at ({grid_point[0]}, {grid_point[1]})")
                                else:
                                    self.failed_constraints.append(
                                        f"Constraint C{idx} (pipe stage {stage_idx} for {stage_instances}): "
                                        f"Cannot find free grid near path position."
                                    )
                                    continue

                            stage_point = Point(
                                (grid_point[0] + 0.5) * self.grid_size,
                                (grid_point[1] + 0.5) * self.grid_size
                            )

                            # Create a group for this stage
                            group_name = f"TVC_PIPE_STAGE_{group_id}"
                            group_id += 1

                            # Allocate a single grid cell for the stage group
                            region = self.allocate_single_grid_region(stage_point)

                            if region:
                                self.placements[group_name] = {
                                    'type': 'group',
                                    'region': region,
                                    'instances': stage_instances,
                                    'constraint_idx': idx,
                                    'is_pipe_stage': True  # Mark as pipe stage for special handling
                                }

                                # Record location for each instance in the stage
                                center_point = region.center()
                                for inst in stage_instances:
                                    self.instance_locations[inst] = center_point
                                    print(f"  → Placed {inst} in pipe stage {group_name} at {center_point}")
                            else:
                                self.failed_constraints.append(
                                    f"Constraint C{idx} (pipe stage {stage_idx} for {stage_instances}): "
                                    f"Cannot allocate grid at position."
                                )
                    else:
                        self.failed_constraints.append(
                            f"Constraint C{idx} (pipe for {constraint.elements}): "
                            f"Cannot find a path from ({start_grid[0]}, {start_grid[1]}) to ({end_grid[0]}, {end_grid[1]}) "
                            f"(path blocked by hard blocks/blockages)."
                        )

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Constraint C{idx} processing error: {str(e)}")
                self.failed_constraints.append(
                    f"Constraint C{idx} processing error for {constraint.elements}: {str(e)}"
                )

            # Update visualization after each constraint
            self._update_placement_visualization(idx)

        # Remove the last constraint text label
        if self.current_constraint_text:
            self.current_constraint_text.remove()
            self.current_constraint_text = None
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def generate_tcl_script(self, filepath: str):
        """
        Generate Innovus TCL script.
        Groups are created for both close_to_target and pipe constraints.
        """
        with open(filepath, 'w') as f:
            f.write("# Pre-placement TCL Script\n")
            f.write("# Generated by Grid-Based Placer\n")
            f.write(f"# Grid Size: {self.grid_size} um\n")

            # Process all groups (from both close_to_target and pipe)
            for name, data in self.placements.items():
                if data['type'] == 'group':
                    region = data['region']
                    f.write(f"deleteInstGroup {name}\n")
                    f.write(f"createInstGroup {name} -region "
                           f"{region.llx:.2f} {region.lly:.2f} "
                           f"{region.urx:.2f} {region.ury:.2f}\n")
                    # Add instances to group
                    instances = " ".join(data['instances'])
                    f.write(f"addInstToInstGroup {name} {{ {instances} }}\n\n")

            # Process single instances (should only be from close_to_target now)
            for name, data in self.placements.items():
                if data['type'] == 'instance':
                    pos = data['position']
                    f.write(f"placeInstance {name} "
                           f"{pos.x:.2f} {pos.y:.2f} -softFixed\n")

    def save_failed_list(self, filepath: str):
        """儲存失敗列表"""
        with open(filepath, 'w') as f:
            if self.failed_constraints:
                for item in self.failed_constraints:
                    f.write(f"{item}\n")
            else:
                f.write("# No failed constraints\n")

    def _draw_static_design_elements(self, ax, show_legend=True):
        """輔助函數：繪製靜態設計元素 (晶片邊界、核心區域、區塊、阻擋、I/O)"""
        # 晶片邊界
        if self.die_boundary:
            die_rect = patches.Rectangle(
                (self.die_boundary.llx, self.die_boundary.lly),
                self.die_boundary.urx - self.die_boundary.llx,
                self.die_boundary.ury - self.die_boundary.lly,
                linewidth=2, edgecolor='black', facecolor='none'
            )
            ax.add_patch(die_rect)

        # 核心區域
        if self.core_area:
            core_rect = patches.Rectangle(
                (self.core_area.llx, self.core_area.lly),
                self.core_area.urx - self.core_area.llx,
                self.core_area.ury - self.core_area.lly,
                linewidth=1.5, edgecolor='blue', facecolor='lightgray', alpha=0.3
            )
            ax.add_patch(core_rect)

        # Blockages (深紅色斜線填充)
        for blockage in self.blockages.values():
            blockage_rect = patches.Rectangle(
                (blockage.boundary.llx, blockage.boundary.lly),
                blockage.boundary.urx - blockage.boundary.llx,
                blockage.boundary.ury - blockage.boundary.lly,
                linewidth=1, edgecolor='darkred',
                facecolor='darkred', alpha=0.5, hatch='//'
            )
            ax.add_patch(blockage_rect)

            cx = blockage.boundary.center().x
            cy = blockage.boundary.center().y

        # 區塊
        for block in self.blocks.values():
            block_rect = patches.Rectangle(
                (block.boundary.llx, block.boundary.lly),
                block.boundary.urx - block.boundary.llx,
                block.boundary.ury - block.boundary.lly,
                linewidth=1, edgecolor='black', facecolor='wheat'
            )
            ax.add_patch(block_rect)
            cx = block.boundary.center().x
            cy = block.boundary.center().y

            # 繪製 Pin Box (深灰色)
            if block.pin_box:
                pin_box_rect = patches.Rectangle(
                    (block.pin_box.llx, block.pin_box.lly),
                    block.pin_box.urx - block.pin_box.llx,
                    block.pin_box.ury - block.pin_box.lly,
                    linewidth=1, edgecolor='dimgray', facecolor='dimgray', alpha=0.6,
                    linestyle='--' # 虛線更容易區分
                )
                ax.add_patch(pin_box_rect)
                pin_cx = block.pin_box.center().x
                pin_cy = block.pin_box.center().y

        # I/O pads
        for name, io in self.io_pads.items():
            io_rect = patches.Rectangle(
                (io.boundary.llx, io.boundary.lly),
                io.boundary.urx - io.boundary.llx,
                io.boundary.ury - io.boundary.lly,
                linewidth=1, edgecolor='green', facecolor='lightgreen', alpha=0.6
            )
            ax.add_patch(io_rect)
            cx = io.boundary.center().x
            cy = io.boundary.center().y

        for name, sensor in self.mp_sensors.items():
            sensor_rect = patches.Rectangle(
                (sensor.boundary.llx, sensor.boundary.lly),
                sensor.boundary.urx - sensor.boundary.llx,
                sensor.boundary.ury - sensor.boundary.lly,
                linewidth=1, edgecolor='orange', facecolor='lightyellow', alpha=0.6
            )
            ax.add_patch(sensor_rect)

        if self.die_boundary:
            ax.set_xlim(self.die_boundary.llx, self.die_boundary.urx)
            ax.set_ylim(self.die_boundary.lly, self.die_boundary.ury)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        if show_legend:
            legend_elements = [
                patches.Patch(color='wheat', label='Hard Blocks'),
                patches.Patch(color='darkred', alpha=0.5, hatch='//', label='Blockages'),
                patches.Patch(color='lightgreen', alpha=0.6, label='I/O Pads'),
                patches.Patch(color='lightyellow', alpha=0.6, label='MP Sensors'),  # FIX 4: 新增圖例項目
                patches.Patch(color='lightgray', alpha=0.3, label='Core Area')
            ]
            # 將圖例放在圖形外部右側，避免遮擋晶片
            ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)

    def visualize_initial_design(self):
        """視覺化原始設計佈局，不含網格和放置結果"""
        if not self.die_boundary:
            print("警告: 晶片邊界未載入，無法視覺化原始設計。")
            return

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        self._draw_static_design_elements(ax)
        ax.set_title('Step 1: Initial Design Layout (After Loading Files)', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (μm)', fontsize=12)
        ax.set_ylabel('Y (μm)', fontsize=12)

        # 添加說明文字 - 放在底部左側，更小的字體和半透明背景
        info_text = f"Die: {self.die_boundary.urx:.0f}x{self.die_boundary.ury:.0f} μm | "
        info_text += f"Core: ({self.core_area.llx:.0f},{self.core_area.lly:.0f})-({self.core_area.urx:.0f},{self.core_area.ury:.0f}) | "
        info_text += f"Blocks: {len(self.blocks)} | Blockages: {len(self.blockages)} | "
        info_text += f"I/O Pads: {len(self.io_pads)} | MP Sensors: {len(self.mp_sensors)}"  # FIX 4: 顯示 MP sensors 數量
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray'))

        plt.tight_layout()
        print("\n>>> Please close the figure window to continue to the next step...")
        plt.show()  # 阻塞模式，等待用戶關閉

    def visualize_grid_state(self):
        """視覺化網格狀態圖 - 優化刻度顯示"""
        if self.grid_map is None:
            print("警告: 網格地圖未建立，無法視覺化網格狀態。")
            return

        fig, ax = plt.subplots(1, 1, figsize=(16, 10))

        # 定義顏色映射
        colors = ['#90EE90', '#FF6B6B', '#FFE66D']  # 綠色(FREE), 紅色(BLOCKED), 黃色(RESERVED)
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        bounds = [0, 1, 2, 3]
        norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        # 顯示網格地圖
        im = ax.imshow(self.grid_map, cmap=cmap, norm=norm, origin='lower',
                      extent=[0, self.grid_width * self.grid_size,
                              0, self.grid_height * self.grid_size],
                      aspect='equal', interpolation='nearest')

        # 優化刻度顯示 - 減少刻度數量
        # X軸：根據寬度決定刻度間隔
        x_max = self.grid_width * self.grid_size
        if x_max <= 1000:
            x_tick_interval = 100  # 每100微米一個刻度
        elif x_max <= 5000:
            x_tick_interval = 250  # 每250微米一個刻度
        elif x_max <= 10000:
            x_tick_interval = 500  # 每500微米一個刻度
        else:
            x_tick_interval = 1000  # 每1000微米一個刻度

        # Y軸：根據高度決定刻度間隔
        y_max = self.grid_height * self.grid_size
        if y_max <= 1000:
            y_tick_interval = 100
        elif y_max <= 5000:
            y_tick_interval = 250
        elif y_max <= 10000:
            y_tick_interval = 500
        else:
            y_tick_interval = 1000

        # 設置主要刻度
        x_ticks = np.arange(0, x_max + 1, x_tick_interval)
        y_ticks = np.arange(0, y_max + 1, y_tick_interval)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        # 添加次要刻度（網格線），但不顯示標籤
        x_minor_ticks = np.arange(0, x_max + 1, self.grid_size)
        y_minor_ticks = np.arange(0, y_max + 1, self.grid_size)
        ax.set_xticks(x_minor_ticks, minor=True)
        ax.set_yticks(y_minor_ticks, minor=True)

        # 設置網格線樣式
        ax.grid(True, which='major', color='black', linewidth=0.8, alpha=0.7)
        ax.grid(True, which='minor', color='gray', linewidth=0.3, alpha=0.3)

        # 旋轉x軸標籤以避免重疊（如果需要）
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

        # 設置標題和標籤
        ax.set_title(f'Step 2: Grid State Map (Grid Size: {self.grid_size} μm)', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (μm)', fontsize=12)
        ax.set_ylabel('Y (μm)', fontsize=12)

        # 添加圖例 - 放在圖形外部右側
        legend_elements = [
            patches.Patch(color='#90EE90', label='FREE - Available for placement'),
            patches.Patch(color='#FF6B6B', label='BLOCKED - Occupied by blocks/blockages/core boundary'),
            patches.Patch(color='#FFE66D', label='RESERVED - Reserved for instance groups')
        ]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.08, 0.5), fontsize=9)

        # 添加統計資訊
        free_count = np.sum(self.grid_map == self.GRID_FREE)
        blocked_count = np.sum(self.grid_map == self.GRID_BLOCKED)
        reserved_count = np.sum(self.grid_map == self.GRID_RESERVED)
        total_count = self.grid_width * self.grid_height

        stats_text = f"Grid Stats: FREE={free_count} ({100*free_count/total_count:.1f}%), "
        stats_text += f"BLOCKED={blocked_count} ({100*blocked_count/total_count:.1f}%), "
        stats_text += f"RESERVED={reserved_count}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

        plt.tight_layout()
        print("\n>>> Please close the figure window to continue to constraint processing...")
        plt.show()  # 阻塞模式，等待用戶關閉

    def _setup_placement_visualization(self):
        """初始化放置結果視覺化的圖形和子圖"""
        if self.die_boundary is None or self.grid_map is None:
            print("錯誤: 晶片邊界或網格地圖未載入，無法設置視覺化。")
            return

        self.fig, (self.ax_grid, self.ax_placement) = plt.subplots(1, 2, figsize=(28, 13))
        self.fig.suptitle('Step 3: Constraint Processing and Placement', fontsize=16, fontweight='bold')

        # --- 左圖：網格地圖 ---
        colors = ['#E0FFE0', '#FFCCCC', '#FFFFCC']  # FREE (淺綠), BLOCKED (淺紅), RESERVED (淺黃)
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        bounds = [0, 1, 2, 3]
        norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        self.grid_img = self.ax_grid.imshow(self.grid_map, cmap=cmap, norm=norm, origin='lower',
                                            extent=[0, self.grid_width * self.grid_size,
                                                    0, self.grid_height * self.grid_size])
        self.ax_grid.set_title('Grid Map (Real-time Update)', fontsize=14)
        self.ax_grid.set_xlabel('X (μm)', fontsize=12)
        self.ax_grid.set_ylabel('Y (μm)', fontsize=12)
        self.ax_grid.grid(True, alpha=0.3)

        # 左圖圖例 - 放在圖形下方外部
        legend_elements_ax1 = [
            patches.Patch(color='#E0FFE0', label='Free (0)'),
            patches.Patch(color='#FFCCCC', label='Blocked (1)'),
            patches.Patch(color='#FFFFCC', label='Reserved (2)')
        ]
        self.ax_grid.legend(handles=legend_elements_ax1, loc='upper center',
                           bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=9)

        # --- 右圖：放置結果 ---
        self._draw_static_design_elements(self.ax_placement, show_legend=False) # 繪製靜態元素作為背景，不顯示重複圖例
        self.ax_placement.set_title('Placement Result with Constraint Visualization', fontsize=14)
        self.ax_placement.set_xlabel('X (μm)', fontsize=12)
        self.ax_placement.set_ylabel('Y (μm)', fontsize=12)

        if USE_NEW_CMAP_API:
            # 新版本 Matplotlib (>= 3.7)
            self.cmap_constraints = plt.colormaps['hsv'].resampled(len(self.constraints) + 1)
        else:
            # 舊版本 Matplotlib (< 3.7)
            self.cmap_constraints = plt.cm.get_cmap('hsv', len(self.constraints) + 1)

        # 設置右圖的圖例 - 放在圖形下方外部，使用兩欄顯示 (更新 Pipe Start 為三角形)
        legend_elements_ax2 = [
            patches.Patch(color='wheat', label='Hard Blocks'), # 重新加入靜態圖例
            patches.Patch(color='darkred', alpha=0.5, hatch='//', label='Blockages'),
            patches.Patch(color='lightgreen', alpha=0.6, label='I/O Pads'),
            patches.Patch(color='lightyellow', alpha=0.6, label='MP Sensors'),  # FIX 4: 新增圖例
            patches.Patch(color='lightgray', alpha=0.3, label='Core Area'),
            patches.Patch(color='gray', alpha=0.3, label='Instance Groups'),
            patches.Patch(color='purple', alpha=0.3, label='Pipe Stage Groups'),  # NEW
            patches.Patch(color='cyan', alpha=0.4, label='SRAM Groups'),  # for single-block SRAM
            patches.Patch(color='magenta', alpha=0.4, label='SRAM Multi-Block Groups'),  # NEW for multi-block
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Single Instances', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='x', color='gray', markersize=10, mew=2, linestyle='None', label='Target Point'),
            plt.Line2D([0], [0], marker='^', color='gray', markersize=8, mew=1, linestyle='None', label='Pipe Start'),  # 改為三角形
            plt.Line2D([0], [0], marker='s', color='gray', markersize=8, mew=1, linestyle='None', label='Pipe End')
        ]
        self.ax_placement.legend(handles=legend_elements_ax2, loc='upper center',
                                bbox_to_anchor=(0.5, -0.08), ncol=7, fontsize=9)

        plt.tight_layout(rect=[0, 0.08, 1, 0.95]) # 調整佈局以容納suptitle和底部圖例
        plt.show(block=False) # 非阻塞模式

    def _update_placement_visualization(self, current_constraint_idx: int):
        """FIX 1: 修復視覺化標籤清理問題"""
        if self.fig is None:
            return

        # 更新網格圖
        self.grid_img.set_data(self.grid_map)

        # 清除舊的放置結果 (包含所有動態添加的元素)
        for artist in self.placement_artists:
            if hasattr(artist, 'remove'):
                artist.remove()
        self.placement_artists.clear()

        # 重新繪製約束目標點/起點/終點
        for idx, constraint in enumerate(self.constraints):
            color = self.cmap_constraints(idx / len(self.constraints))

            if constraint.type == 'close_to_target':
                # Draw bounding rectangle for sram_group - NEW
                if constraint.target_type == 'sram_group' and constraint.vis_bounding_rect:
                    bounding_rect = constraint.vis_bounding_rect
                    bound_rect_patch = patches.Rectangle(
                        (bounding_rect.llx, bounding_rect.lly),
                        bounding_rect.urx - bounding_rect.llx,
                        bounding_rect.ury - bounding_rect.lly,
                        linewidth=2, edgecolor=color,
                        facecolor='none', linestyle='--', zorder=8
                    )
                    self.ax_placement.add_patch(bound_rect_patch)
                    self.placement_artists.append(bound_rect_patch)

                    # Add label for bounding rect
                    artist_text = self.ax_placement.text(
                        bounding_rect.center().x, bounding_rect.ury + 10,
                        f'C{idx} Bounding', ha='center', va='bottom',
                        fontsize=8, color=color,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'),
                        zorder=10
                    )
                    self.placement_artists.append(artist_text)

                if constraint.vis_target_point:
                    target_point = constraint.vis_target_point
                    artist = self.ax_placement.plot(target_point.x, target_point.y, 'x', color=color, markersize=10, mew=2, zorder=10)[0]
                    self.placement_artists.append(artist)
                    artist_text = self.ax_placement.annotate(f'C{idx} Target', (target_point.x, target_point.y), fontsize=7, color=color,
                                     xytext=(5, 5), textcoords='offset points', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'), zorder=10)
                    self.placement_artists.append(artist_text)
            elif constraint.type == 'pipe':
                if constraint.vis_start_point:
                    artist = self.ax_placement.plot(constraint.vis_start_point.x, constraint.vis_start_point.y, '^', color=color, markersize=8, mew=1, zorder=10)[0]
                    self.placement_artists.append(artist)
                    artist_text = self.ax_placement.annotate(f'C{idx} Start', (constraint.vis_start_point.x, constraint.vis_start_point.y), fontsize=7, color=color,
                                         xytext=(5, 5), textcoords='offset points', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'), zorder=10)
                    self.placement_artists.append(artist_text)

                if constraint.vis_end_point:
                    artist = self.ax_placement.plot(constraint.vis_end_point.x, constraint.vis_end_point.y, 's', color=color, markersize=8, mew=1, zorder=10)[0]
                    self.placement_artists.append(artist)
                    artist_text = self.ax_placement.annotate(f'C{idx} End', (constraint.vis_end_point.x, constraint.vis_end_point.y), fontsize=7, color=color,
                                         xytext=(5, 5), textcoords='offset points', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'), zorder=10)
                    self.placement_artists.append(artist_text)

                # 繪製 pipe 路徑 (如果存在) - 使用實際保存的路徑
                if constraint.vis_pipe_path:
                    path = constraint.vis_pipe_path
                    path_coords_x = [(p[0] + 0.5) * self.grid_size for p in path]
                    path_coords_y = [(p[1] + 0.5) * self.grid_size for p in path]
                    line, = self.ax_placement.plot(path_coords_x, path_coords_y, color=color, linestyle='--', linewidth=1, alpha=0.7, zorder=9)
                    self.placement_artists.append(line)

        # 重新繪製放置結果
        for name, data in self.placements.items():
            constraint_idx = data.get('constraint_idx', -1)
            color = self.cmap_constraints(constraint_idx / len(self.constraints)) if constraint_idx != -1 else 'black'

            if data['type'] == 'group':
                region = data['region']
                # Different style for pipe stage groups and SRAM groups
                is_pipe_stage = data.get('is_pipe_stage', False)
                is_sram = data.get('is_sram', False)
                is_sram_group = data.get('is_sram_group', False)  # NEW

                if is_sram_group:  # NEW: SRAM multi-block groups - magenta color
                    group_rect = patches.Rectangle(
                        (region.llx, region.lly),
                        region.urx - region.llx,
                        region.ury - region.lly,
                        linewidth=2.5, edgecolor=color,
                        facecolor='magenta', alpha=0.4, zorder=5,
                        linestyle='-'  # Solid line
                    )
                elif is_sram:
                    # SRAM groups - cyan color with solid line
                    group_rect = patches.Rectangle(
                        (region.llx, region.lly),
                        region.urx - region.llx,
                        region.ury - region.lly,
                        linewidth=2.5, edgecolor=color,
                        facecolor='cyan', alpha=0.4, zorder=5,
                        linestyle='-'  # Solid line for SRAM
                    )
                elif is_pipe_stage:
                    # Pipe stage groups - smaller, different color scheme
                    group_rect = patches.Rectangle(
                        (region.llx, region.lly),
                        region.urx - region.llx,
                        region.ury - region.lly,
                        linewidth=2, edgecolor=color,
                        facecolor='purple', alpha=0.4, zorder=5,
                        linestyle=':'  # Dotted line for pipe stages
                    )
                else:
                    # Regular groups
                    group_rect = patches.Rectangle(
                        (region.llx, region.lly),
                        region.urx - region.llx,
                        region.ury - region.lly,
                        linewidth=2, edgecolor=color,
                        facecolor=color, alpha=0.3, zorder=5
                    )

                self.ax_placement.add_patch(group_rect)
                self.placement_artists.append(group_rect)

                cx = region.center().x
                cy = region.center().y
                artist_text = self.ax_placement.text(cx, cy, name, ha='center', va='center',
                        fontsize=9, color='black', weight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'), zorder=6)
                self.placement_artists.append(artist_text)

            elif data['type'] == 'instance':
                pos = data['position']
                artist = self.ax_placement.scatter(pos.x, pos.y, color=color, s=50, zorder=7, edgecolors='black', linewidth=0.5)
                self.placement_artists.append(artist)
                artist_text = self.ax_placement.annotate(name, (pos.x, pos.y), fontsize=7, color='black',
                           xytext=(5, 5), textcoords='offset points', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'), zorder=8)
                self.placement_artists.append(artist_text)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events() # 刷新事件，確保圖形更新

    def run(self, tvc_json: str, constraints: str,
            output_tcl: str = 'dft_regs_pre_place.tcl',
            failed_list: str = 'failed_preplace.list'):
        """執行完整流程"""
        print("=" * 60)
        print("Grid-Based Pre-placement Tool with Blockage Support")
        print(f"Grid Size: {self.grid_size} μm")
        print(f"Root Module Name: {self.root_module_name}")
        print("=" * 60)

        # 1. 載入檔案
        print("\n[1] Loading input files...")
        self.load_tvc_json(tvc_json)
        print(f"    ✓ Loaded design with {len(self.blocks)} blocks")
        print(f"    ✓ Loaded {len(self.blockages)} blockages")
        print(f"    ✓ Loaded {len(self.io_pads)} I/O pads from TVC JSON")
        print(f"    ✓ Loaded {len(self.mp_sensors)} MP sensors from TVC JSON")

        self.load_constraints(constraints)
        print(f"    ✓ Loaded {len(self.constraints)} constraints")

        # Count constraint types
        close_count = sum(1 for c in self.constraints if c.type == 'close_to_target')
        pipe_count = sum(1 for c in self.constraints if c.type == 'pipe')
        sram_count = sum(1 for c in self.constraints if c.type == 'close_to_target' and c.target_type == 'sram')
        sram_group_count = sum(1 for c in self.constraints if c.type == 'close_to_target' and c.target_type == 'sram_group')  # NEW
        print(f"    ✓ Constraint types: {close_count} close_to_target ({sram_count} SRAM, {sram_group_count} SRAM_GROUP), {pipe_count} pipe")
        print(f"    ✓ Processing order: close_to_target first, then pipe")
        print(f"    ✓ Pipe constraints will create stage groups (one grid per stage)")
        if sram_count > 0:
            print(f"    ✓ SRAM constraints will place groups adjacent to target block pin edges")
        if sram_group_count > 0:  # NEW
            print(f"    ✓ SRAM_GROUP constraints will find max rectangle within bounding box of multiple blocks")

        # 視覺化原始設計佈局 (步驟 1)
        print("\n[VISUALIZATION] Step 1: Showing initial design layout...")
        self.visualize_initial_design()

        # 2. 建立網格
        print("\n[2] Building grid map...")
        self.build_grid_map()
        print(f"    ✓ Grid dimensions: {self.grid_width} x {self.grid_height} grids")

        # 統計網格狀態
        free_grids = np.sum(self.grid_map == self.GRID_FREE)
        blocked_grids = np.sum(self.grid_map == self.GRID_BLOCKED)
        reserved_grids = np.sum(self.grid_map == self.GRID_RESERVED)
        total_grids = self.grid_width * self.grid_height

        print(f"    ✓ Initial Free grids: {free_grids}/{total_grids} "
              f"({100*free_grids/total_grids:.1f}%)")
        print(f"    ✓ Initial Blocked grids: {blocked_grids} grids")

        # 視覺化網格狀態 (步驟 2)
        print("\n[VISUALIZATION] Step 2: Showing grid state map...")
        self.visualize_grid_state()

        # 3. 處理約束
        print("\n[3] Processing constraints with live visualization...")
        print("    (Step 3: Dynamic constraint processing visualization)")
        # 在處理約束前，先設置好動態視覺化圖形
        self._setup_placement_visualization()
        self.process_constraints()

        # 處理完所有約束後，讓圖形保持顯示
        print("\n[VISUALIZATION] Final placement results")
        # 移除最後的約束標籤
        if self.current_constraint_text:
            self.current_constraint_text.remove()
            self.current_constraint_text = None
        self.fig.suptitle('Final Placement Results', fontsize=16, fontweight='bold')
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        print("\n>>> Please close the figure window to generate output files...")
        plt.show()  # 阻塞模式，讓使用者檢查最終結果

        # 4. 產生輸出
        print("\n[4] Generating outputs...")
        self.generate_tcl_script(output_tcl)
        print(f"    ✓ TCL script: {output_tcl}")

        self.save_failed_list(failed_list)
        print(f"    ✓ Failed list: {failed_list}")

        # 統計最終結果
        print("\n[5] Summary:")
        successful_placements = len(self.placements)
        total_elements = sum(len(c.elements) for c in self.constraints)
        total_instances_placed = len(self.instance_locations)
        print(f"    ✓ Successfully placed: {successful_placements} items (groups and instances)")
        print(f"    ✓ Total individual instances placed: {total_instances_placed}")
        print(f"    ✓ Failed constraints: {len(self.failed_constraints)}")

        print("\n" + "=" * 60)
        print("✓ Complete! All processing finished.")
        print("=" * 60)




if __name__ == "__main__":
    import sys

    # 預設使用新的根模組名稱
    ROOT_MODULE_NAME = "SoIC_A16_eTV5_root"

    if len(sys.argv) == 3:
        placer = GridBasedPlacer(grid_size=10.0, root_module_name=ROOT_MODULE_NAME)
        placer.run(
            tvc_json=sys.argv[1],
            constraints=sys.argv[2]
        )
    else:
        print("Usage: python placer.py <tvc.json> <constraints.phy>")
        print(f"\nRunning test mode with blockages and default root module: '{ROOT_MODULE_NAME}'...")

        # 建立測試檔案
        create_test_files_with_blockages()

        # 執行測試
        placer = GridBasedPlacer(grid_size=50.0, root_module_name=ROOT_MODULE_NAME)
        placer.run(
            tvc_json='test_tvc_blockage.json',
            constraints='test_constraints.phy'
        )
