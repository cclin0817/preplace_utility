#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Grid-Based Pre-placement Tool with Blockage Support
使用網格結構和 A* 演算法的預放置工具
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
import heapq
import re
import time # 引入 time 模組用於暫停

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

@dataclass
class Constraint:
    """放置約束"""
    type: str  # 'close_to_target', 'pipe'
    elements: List[str] = field(default_factory=list) # 受影響的元件列表

    # close_to_target 相關
    target_type: Optional[str] = None # 'cell', 'coords', 'pin'
    target_name: Optional[str] = None # 目標名稱 (cell)
    target_coords: Optional[Point] = None # 目標座標 (coords)

    # pipe 相關
    start: Optional[str] = None   # pipe 起點元件
    end: Optional[str] = None     # pipe 終點元件

    element_type: str = 'single'  # 'single', 'instancesgroup'
    area: float = 20.0            # 僅用於 instancesgroup

    # 視覺化輔助
    vis_target_point: Optional[Point] = None # close_to_target 實際解析出的目標點
    vis_start_point: Optional[Point] = None  # pipe 實際解析出的起點
    vis_end_point: Optional[Point] = None    # pipe 實際解析出的終點

    def __str__(self):
        if self.type == 'close_to_target':
            target_info = ""
            if self.target_type == 'coords' and self.target_coords:
                target_info = f"coords={self.target_coords}"
            elif self.target_type in ['cell', 'pin'] and self.target_name:
                target_info = f"{self.target_type}='{self.target_name}'"
            return f"Constraint(Type='{self.type}', Target={target_info})"
        elif self.type == 'pipe':
            return f"Constraint(Type='{self.type}', Start='{self.start}', End='{self.end}')"
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
        self.ip_dimensions: Dict[str, Tuple[float, float]] = {}

        # 網格地圖
        self.grid_map: Optional[np.ndarray] = None
        self.grid_width: int = 0
        self.grid_height: int = 0

        # 約束和放置結果
        self.constraints: List[Constraint] = []
        self.placements: Dict[str, any] = {} # 儲存實際放置的元件或群組資訊
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

    def load_tvc_json(self, filepath: str):
        """載入 TVC JSON 檔案"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        if 'IPS' in data:
            for ip_name, ip_data in data['IPS'].items():
                if 'SIZE_X' in ip_data and 'SIZE_Y' in ip_data:
                    self.ip_dimensions[ip_name] = (float(ip_data['SIZE_X']), float(ip_data['SIZE_Y']))
            print(f"    ✓ Loaded {len(self.ip_dimensions)} IP dimensions.")

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

                    if cell_name in self.ip_dimensions:
                        size_x, size_y = self.ip_dimensions[cell_name]
                        llx, lly = location[0], location[1]
                        urx, ury = llx + size_x, lly + size_y

                        sensor_block = Block(
                            name=name,
                            boundary=Rectangle(llx, lly, urx, ury)
                        )
                        self.blocks[name] = sensor_block
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

                    if cell_name in self.ip_dimensions:
                        size_x, size_y = self.ip_dimensions[cell_name]
                        llx, lly = location[0], location[1]
                        urx, ury = llx + size_x, lly + size_y

                        self.io_pads[name] = IOPad(
                            name=name,
                            boundary=Rectangle(llx, lly, urx, ury)
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
                elif 'pipe' in line: # 新的約束類型名稱
                    constraint.type = 'pipe'
                    constraint.element_type = 'single'  # pipe 類型固定為 single

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
                            else: # 'cell' 或 'pin'
                                constraint.target_name = target_value
                        elif line.startswith('Element Type:'):
                            constraint.element_type = line.split(':')[1].strip()
                        elif line.startswith('Area:'):
                            constraint.area = float(line.split(':')[1].strip())
                        elif line.startswith('Element:'):
                            constraint.elements = line.split(':')[1].strip().split()

                    # pipe 相關解析
                    elif constraint.type == 'pipe':
                        if line.startswith('Start:'):
                            constraint.start = line.split(':')[1].strip()
                        elif line.startswith('End:'):
                            constraint.end = line.split(':')[1].strip()
                        elif line.startswith('Stage'): # Stage1, Stage2, ...
                            elements = line.split(':')[1].strip().split()
                            constraint.elements.extend(elements) # 累積所有 Stage 的元件
                        # 跳過 Element Type 和 Area 行，因為 pipe 類型固定為 single

                    i += 1

                if constraint.elements: # 只有當有元件時才加入約束列表
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
                if (0 <= nx < self.grid_width and 0 <= ny < self.grid_height and (nx, ny) not in visited):
                    queue.append((nx, ny))

        return None # 找不到可用網格

    def a_star_path(self, start: Tuple[int, int],
                    end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        使用 A* 演算法尋找路徑，避開 GRID_BLOCKED 和 GRID_RESERVED。
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

                # 檢查邊界和障礙 (GRID_BLOCKED)
                if (0 <= nx < self.grid_width and
                    0 <= ny < self.grid_height and
                    self.grid_map[ny, nx] == self.GRID_FREE and # 只能走 GRID_FREE
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
        elif constraint.target_type == 'cell':
            # 對於 cell 類型，直接查找
            if constraint.target_name in self.blocks:
                block = self.blocks[constraint.target_name]
                # cell 類型通常使用 boundary 中心，除非特別需要 pin_box
                return block.boundary.center()
            elif constraint.target_name in self.io_pads:
                    return self.io_pads[constraint.target_name].boundary.center()
        return None

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

                # 檢查邊界
                if not (0 <= ll_grid_x < ur_grid_x <= self.grid_width and
                        0 <= ll_grid_y < ur_grid_y <= self.grid_height):
                    continue

                is_free = True
                for y_grid in range(ll_grid_y, ur_grid_y):
                    for x_grid in range(ll_grid_x, ur_grid_x):
                        if not (0 <= x_grid < self.grid_width and 0 <= y_grid < self.grid_height):
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
                        # 如果找到的區域非常接近目標點，可以提前返回
                        if dist_sq < (self.grid_size * 2)**2: # 距離目標點 2 個 grid_size 內
                            # 標記區域為已保留
                            for y_grid in range(ll_grid_y, ur_grid_y):
                                for x_grid in range(ll_grid_x, ur_grid_x):
                                    self.grid_map[y_grid, x_grid] = self.GRID_RESERVED
                            return best_region_rect
            if best_region_rect: # 如果當前 radius 找到了至少一個 region，就不再擴大 radius 搜尋
                break # 確保找到的是最靠近的，而不是最遠的

        if best_region_rect:
            # FIXED: 使用正確的 to_grid_region 方法，返回 4 個值
            llx_grid, lly_grid, urx_grid, ury_grid = best_region_rect.to_grid_region(self.grid_size)

            # 標記區域為已保留
            for y_grid in range(lly_grid, ury_grid):
                for x_grid in range(llx_grid, urx_grid):
                    if 0 <= x_grid < self.grid_width and 0 <= y_grid < self.grid_height:
                        self.grid_map[y_grid, x_grid] = self.GRID_RESERVED

        return best_region_rect


    def process_constraints(self, debug_plot_interval: float = 0.01):
        """處理所有約束，並在每次處理後更新視覺化"""
        group_id = 0

        # 初始化視覺化 (如果尚未初始化)
        if self.fig is None:
            self._setup_placement_visualization()

        for idx, constraint in enumerate(self.constraints):
            #print(f"處理約束 C{idx}: Type={constraint.type}, Elements={constraint.elements}")
            print(f"處理約束 C{idx}: {constraint}")
            # 更新當前約束的文字標籤 - 放在上方中央避免遮擋
            if self.current_constraint_text:
                self.current_constraint_text.remove()
            self.current_constraint_text = self.ax_placement.text(
                0.5, 0.98, f"Processing C{idx}: {constraint.type}",
                transform=self.ax_placement.transAxes, va='top', ha='center',
                fontsize=12, color='red', bbox=dict(facecolor='yellow', alpha=0.9, edgecolor='red', boxstyle='round,pad=0.3')
            )

            try:
                if constraint.type == 'close_to_target':
                    target_point = self.get_target_point_for_constraint(constraint)
                    print('target_point', target_point)
                    if not target_point:
                        self.failed_constraints.append(
                            f"Constraint C{idx} for elements {constraint.elements}: Target '{constraint.target_name or str(constraint.target_coords)}' not found or invalid."
                        )
                        self.ax_placement.text(target_point.x, target_point.y, "Failed!", color='red', fontsize=8, ha='center', va='center')
                        self._update_placement_visualization(idx) # 更新視覺化
                        time.sleep(debug_plot_interval)
                        continue
                    constraint.vis_target_point = target_point # 儲存用於視覺化

                    if constraint.element_type == 'instancesgroup':
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
                        else:
                            self.failed_constraints.append(
                                f"Constraint C{idx} ({group_name}, elements: {constraint.elements}): Cannot allocate region for area {constraint.area} (no free space found)."
                            )
                    else: # single instance
                        position_grid = self.find_nearest_free_grid(target_point)

                        if position_grid:
                            # 將找到的網格標記為 BLOCKED，避免其他單一實例重複佔用
                            self.grid_map[position_grid[1], position_grid[0]] = self.GRID_BLOCKED

                            position = Point(
                                (position_grid[0] + 0.5) * self.grid_size,
                                (position_grid[1] + 0.5) * self.grid_size
                            )
                            for element in constraint.elements:
                                self.placements[element] = {
                                    'type': 'instance',
                                    'position': position,
                                    'constraint_idx': idx
                                }
                        else:
                            for element in constraint.elements:
                                self.failed_constraints.append(
                                    f"Constraint C{idx} ({element}): Cannot find placement near target "
                                    f"'{constraint.target_name or str(constraint.target_coords)}' (no free space found)."
                                )

                elif constraint.type == 'pipe':
                    start_pos = None
                    end_pos = None

                    # 嘗試從已放置的元件中獲取起點和終點
                    if constraint.start in self.placements and 'position' in self.placements[constraint.start]:
                        start_pos = self.placements[constraint.start]['position']
                    elif constraint.start in self.blocks:
                        start_pos = self.blocks[constraint.start].boundary.center()
                    elif constraint.start in self.io_pads:
                        start_pos = self.io_pads[constraint.start].boundary.center()

                    if constraint.end in self.placements and 'position' in self.placements[constraint.end]:
                        end_pos = self.placements[constraint.end]['position']
                    elif constraint.end in self.blocks:
                        end_pos = self.blocks[constraint.end].boundary.center()
                    elif constraint.end in self.io_pads:
                        end_pos = self.io_pads[constraint.end].boundary.center()

                    # 如果起點或終點未被找到，則使用 Core Area 內的預設點
                    if not start_pos:
                        # 嘗試在 core_area 內找一個 free grid
                        start_grid_candidate = self.find_nearest_free_grid(Point(
                            self.core_area.llx + self.grid_size,
                            self.core_area.lly + self.grid_size
                        ))
                        if start_grid_candidate:
                            start_pos = Point(
                                (start_grid_candidate[0] + 0.5) * self.grid_size,
                                (start_grid_candidate[1] + 0.5) * self.grid_size
                            )
                        else: # 如果連 core area 內都找不到 free grid
                            start_pos = Point(
                                self.core_area.llx + self.grid_size,
                                self.core_area.lly + self.grid_size
                            )
                        print(f"警告: Constraint C{idx}: Start element '{constraint.start}' not found or placed, using a point near ({start_pos.x:.2f}, {start_pos.y:.2f}).")

                    if not end_pos:
                        # 嘗試在 core_area 內找一個 free grid
                        end_grid_candidate = self.find_nearest_free_grid(Point(
                            self.core_area.urx - self.grid_size,
                            self.core_area.ury - self.grid_size
                        ))
                        if end_grid_candidate:
                            end_pos = Point(
                                (end_grid_candidate[0] + 0.5) * self.grid_size,
                                (end_grid_candidate[1] + 0.5) * self.grid_size
                            )
                        else: # 如果連 core area 內都找不到 free grid
                            end_pos = Point(
                                self.core_area.urx - self.grid_size,
                                self.core_area.ury - self.grid_size
                            )
                        print(f"警告: Constraint C{idx}: End element '{constraint.end}' not found or placed, using a point near ({end_pos.x:.2f}, {end_pos.y:.2f}).")

                    constraint.vis_start_point = start_pos
                    constraint.vis_end_point = end_pos

                    start_grid = self.find_nearest_free_grid(start_pos)
                    end_grid = self.find_nearest_free_grid(end_pos)

                    if not start_grid or not end_grid:
                        self.failed_constraints.append(
                            f"Constraint C{idx} (pipe for {constraint.elements}): Could not find free start or end grid."
                        )
                        self._update_placement_visualization(idx) # 更新視覺化
                        time.sleep(debug_plot_interval)
                        continue

                    path = self.a_star_path(start_grid, end_grid)

                    if path:
                        num_elements = len(constraint.elements)
                        if num_elements == 0:
                            self._update_placement_visualization(idx) # 更新視覺化
                            time.sleep(debug_plot_interval)
                            continue

                        positions = []
                        # 確保至少有 num_elements + 1 個間隔，讓元件可以分散
                        # 如果路徑太短，讓每個元件盡可能分散
                        path_step = max(1, len(path) // (num_elements + 1))

                        for i in range(num_elements):
                            idx_on_path = min((i + 1) * path_step, len(path) - 1)
                            grid_point = path[idx_on_path]

                            # 將路徑上的網格點標記為 BLOCKED
                            if self.grid_map[grid_point[1], grid_point[0]] == self.GRID_FREE:
                                self.grid_map[grid_point[1], grid_point[0]] = self.GRID_BLOCKED

                            positions.append(Point(
                                (grid_point[0] + 0.5) * self.grid_size,
                                (grid_point[1] + 0.5) * self.grid_size
                            ))

                        for element, pos in zip(constraint.elements, positions):
                            self.placements[element] = {
                                'type': 'instance',
                                'position': pos,
                                'constraint_idx': idx
                            }
                    else:
                        self.failed_constraints.append(
                            f"Constraint C{idx} (pipe for {constraint.elements}): Cannot find a path from {start_grid} to {end_grid} (no free path)."
                        )

            except Exception as e:
                #self.failed_constraints.append(f"Constraint C{idx} {constraint.type} processing error for {constraint.elements}: {str(e)}")
                import traceback
                traceback.print_exc()  # 會顯示完整的錯誤堆疊
                print(f"Constraint C{idx} processing error: {str(e)}")
                self.failed_constraints.append(f"Constraint C{idx} processing error for {constraint.elements}: {str(e)}")

            # 每處理一個約束就更新視覺化
            self._update_placement_visualization(idx)
            time.sleep(debug_plot_interval) # 暫停一段時間以便觀察

        # 移除最後一個約束文字標籤
        if self.current_constraint_text:
            self.current_constraint_text.remove()
            self.current_constraint_text = None
        self.fig.canvas.draw_idle() # 確保最後的更新顯示出來
        self.fig.canvas.flush_events()


    def generate_tcl_script(self, filepath: str):
        """產生 Innovus TCL 腳本"""
        with open(filepath, 'w') as f:
            f.write("# Pre-placement TCL Script\n")
            f.write("# Generated by Grid-Based Placer\n")
            f.write(f"# Grid Size: {self.grid_size} um\n")
            f.write(f"# Blockages: {len(self.blockages)}\n\n")

            # 先處理群組，因為群組需要先定義才能 addInstToInstGroup
            for name, data in self.placements.items():
                if data['type'] == 'group':
                    region = data['region']
                    f.write(f"deleteInstGroup {name}\n")
                    f.write(f"createInstGroup {name} -region "
                           f"{region.llx:.2f} {region.lly:.2f} "
                           f"{region.urx:.2f} {region.ury:.2f}\n")
                    # 將實例加入群組 (確保這些實例在 Innovus 中是存在的)
                    instances = " ".join(data['instances'])
                    f.write(f"addInstToInstGroup {name} {{ {instances} }}\n\n")

            # 再處理單一實例
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
            #ax.text(cx, cy, blockage.name, ha='center', va='center',
            #        fontsize=8, color='white', weight='bold')

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
            #ax.text(cx, cy, block.name, ha='center', va='center', fontsize=8)

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
                # 可選：在 Pin Box 中心標註
                pin_cx = block.pin_box.center().x
                pin_cy = block.pin_box.center().y
                #ax.text(pin_cx, pin_cy, 'Pin', ha='center', va='center', fontsize=6, color='white', alpha=0.8)


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
            #ax.text(cx, cy, name, ha='center', va='center', fontsize=8, color='darkgreen')

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
        info_text += f"Blocks: {len(self.blocks)} | Blockages: {len(self.blockages)} | I/O Pads: {len(self.io_pads)}"
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray'))

        plt.tight_layout()
        print("\n>>> Please close the figure window to continue to the next step...")
        plt.show()  # 阻塞模式，等待用戶關閉

    def visualize_grid_state(self):
        """視覺化網格狀態圖"""
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

        # 添加網格線
        ax.set_xticks(np.arange(0, self.grid_width * self.grid_size + 1, self.grid_size))
        ax.set_yticks(np.arange(0, self.grid_height * self.grid_size + 1, self.grid_size))
        ax.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.5)

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

        # 初始化約束顏色映射
        self.cmap_constraints = plt.cm.get_cmap('hsv', len(self.constraints) + 1)

        # 設置右圖的圖例 - 放在圖形下方外部，使用兩欄顯示
        legend_elements_ax2 = [
            patches.Patch(color='wheat', label='Hard Blocks'), # 重新加入靜態圖例
            patches.Patch(color='darkred', alpha=0.5, hatch='//', label='Blockages'),
            patches.Patch(color='lightgreen', alpha=0.6, label='I/O Pads'),
            patches.Patch(color='lightgray', alpha=0.3, label='Core Area'),
            patches.Patch(color='gray', alpha=0.3, label='Instance Groups'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Single Instances', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='x', color='gray', markersize=10, mew=2, linestyle='None', label='Target Point'),
            plt.Line2D([0], [0], marker='o', color='gray', markersize=8, mew=1, linestyle='None', label='Pipe Start'),
            plt.Line2D([0], [0], marker='s', color='gray', markersize=8, mew=1, linestyle='None', label='Pipe End')
        ]
        self.ax_placement.legend(handles=legend_elements_ax2, loc='upper center',
                                bbox_to_anchor=(0.5, -0.08), ncol=5, fontsize=9)

        plt.tight_layout(rect=[0, 0.08, 1, 0.95]) # 調整佈局以容納suptitle和底部圖例
        plt.show(block=False) # 非阻塞模式

    def _update_placement_visualization(self, current_constraint_idx: int):
        """更新放置結果視覺化圖形"""
        if self.fig is None:
            return

        # 更新網格圖
        self.grid_img.set_data(self.grid_map)

        # 清除舊的放置結果 (動態添加的元素)
        for artist in self.placement_artists:
            artist.remove()
        self.placement_artists.clear()

        # 重新繪製約束目標點/起點/終點
        for idx, constraint in enumerate(self.constraints):
            color = self.cmap_constraints(idx / len(self.constraints))

            if constraint.type == 'close_to_target' and constraint.vis_target_point:
                target_point = constraint.vis_target_point
                artist = self.ax_placement.plot(target_point.x, target_point.y, 'x', color=color, markersize=10, mew=2, zorder=10)[0]
                self.placement_artists.append(artist)
                artist_text = self.ax_placement.annotate(f'C{idx} Target', (target_point.x, target_point.y), fontsize=7, color=color,
                                     xytext=(5, 5), textcoords='offset points', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'), zorder=10)
                #self.placement_artists.append(artist_text)
            elif constraint.type == 'pipe':
                if constraint.vis_start_point:
                    artist = self.ax_placement.plot(constraint.vis_start_point.x, constraint.vis_start_point.y, 'o', color=color, markersize=8, mew=1, zorder=10)[0]
                    self.placement_artists.append(artist)
                    artist_text = self.ax_placement.annotate(f'C{idx} Start', (constraint.vis_start_point.x, constraint.vis_start_point.y), fontsize=7, color=color,
                                         xytext=(5, 5), textcoords='offset points', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'), zorder=10)
                    #self.placement_artists.append(artist_text)
                if constraint.vis_end_point:
                    artist = self.ax_placement.plot(constraint.vis_end_point.x, constraint.vis_end_point.y, 's', color=color, markersize=8, mew=1, zorder=10)[0]
                    self.placement_artists.append(artist)
                    artist_text = self.ax_placement.annotate(f'C{idx} End', (constraint.vis_end_point.x, constraint.vis_end_point.y), fontsize=7, color=color,
                                         xytext=(5, 5), textcoords='offset points', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'), zorder=10)
                    #self.placement_artists.append(artist_text)

                # 繪製 pipe 路徑 (如果存在)
                if constraint.vis_start_point and constraint.vis_end_point:
                    start_grid = constraint.vis_start_point.to_grid(self.grid_size)
                    end_grid = constraint.vis_end_point.to_grid(self.grid_size)
                    path = self.a_star_path(start_grid, end_grid)
                    if path:
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
                #self.placement_artists.append(artist_text)

            elif data['type'] == 'instance':
                pos = data['position']
                artist = self.ax_placement.scatter(pos.x, pos.y, color=color, s=50, zorder=7, edgecolors='black', linewidth=0.5)
                self.placement_artists.append(artist)
                artist_text = self.ax_placement.annotate(name, (pos.x, pos.y), fontsize=7, color='black',
                           xytext=(5, 5), textcoords='offset points', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'), zorder=8)
                #self.placement_artists.append(artist_text)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events() # 刷新事件，確保圖形更新

    def run(self, tvc_json: str, constraints: str,
            output_tcl: str = 'dft_regs_pre_place.tcl',
            failed_list: str = 'failed_preplace.list',
            debug_plot_interval: float = 0.5): # 新增參數控制更新間隔
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

        self.load_constraints(constraints)
        print(f"    ✓ Loaded {len(self.constraints)} constraints")

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
        self.process_constraints(debug_plot_interval) # 傳入間隔時間

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
        print(f"    ✓ Successfully placed: {successful_placements} items")
        print(f"    ✓ Failed constraints: {len(self.failed_constraints)}")

        print("\n" + "=" * 60)
        print("✓ Complete! All processing finished.")
        print("=" * 60)


# 測試用的範例 JSON 檔案
def create_test_files_with_blockages():
    """建立包含 blockages 的測試檔案"""
    import json

    # TVC JSON with blockages
    tvc_data = {
        "S1_output": {
            "SoIC_A16_eTV5_root": { # 修改為新的根模組名稱
                "TOPShape": {
                    "Name": "test_chip",
                    "DieCoords": [[0, 0], [3000, 0], [3000, 2000], [0, 2000]],
                    "CoreCoords": [[200, 200], [2800, 200], [2800, 1800], [200, 1800]]
                },
                "BlockShapes": [{
                    "BlockName": "u_block1",
                    "DesignName": "BLOCK1",
                    "Coords": [[500, 500], [1000, 500], [1000, 1000], [500, 1000]],
                    "PinCoords": [[[600, 600], [650, 650]], [[700,700],[750,750]]], # 測試 pin box 列表
                    "Attribute": "Module"
                }, {
                    "BlockName": "u_block2",
                    "DesignName": "BLOCK2",
                    "Coords": [[2000, 1200], [2500, 1200], [2500, 1700], [2000, 1700]],
                    "Attribute": "Module"
                }],
                "Blockages": [
                    # Blockage 1: 在核心區域中間
                    [[1500, 800], [1700, 800], [1700, 1200], [1500, 1200], [1500, 800]],
                    # Blockage 2: 在右側
                    [[2200, 500], [2400, 500], [2400, 700], [2200, 700], [2200, 500]],
                    # Blockage 3: 靠近u_block1
                    [[1100, 600], [1200, 600], [1200, 700], [1100, 700]]
                ]
            }
        },
        # 新增 Solution_GPIO 區塊
        "Solution_GPIO": {
            "SoIC_A16_eTV5_root": {
                "GPIO": [
                    {
                        "Name": "u_io_pad/PVDD01_0",
                        "Location": [0.0, 996.06],
                        "Orientation": "MY",
                        "CellName": "PVDD1CODCDGM_H",
                        "Legend_key": "PVDD1CODCDGM_H"
                    },
                    {
                        "Name": "u_io_pad/PVDD08_1",
                        "Location": [0.0, 1048.97],
                        "Orientation": "R180",
                        "CellName": "PVDD08CODCDGM_H",
                        "Legend_key": "PVDD08CODCDGM_H"
                    },
                    {
                        "Name": "u_io_pad/PVDD1204_12_3",
                        "Location": [0.0, 1152.06],
                        "Orientation": "MY",
                        "CellName": "PVDD1204CODCDGM_H",
                        "Legend_key": "PVDD1204CODCDGM_H"
                    },
                    {
                        "Name": "u_io_pad/VDD_CORNER_TOP_RIGHT",
                        "Location": [2970.0, 1970.0], # 模擬右上角 I/O Pad
                        "Orientation": "R90",
                        "CellName": "VDD_CORNER",
                        "Legend_key": "VDD_CORNER"
                    }
                ]
            }
        },
        # 新增 IPS 區塊以提供 I/O Pad 的尺寸資訊
        "IPS": {
            "PVDD1CODCDGM_H": {"SIZE_X": 30.0, "SIZE_Y": 52.91},
            "PVDD08CODCDGM_H": {"SIZE_X": 30.0, "SIZE_Y": 52.91},
            "PVDD1204CODCDGM_H": {"SIZE_X": 30.0, "SIZE_Y": 52.91},
            "VDD_CORNER": {"SIZE_X": 30.0, "SIZE_Y": 30.0}, # 假設角落 I/O Pad 尺寸
            # 也可以包含 Blocks 的 IP 尺寸
            "BLOCK1": {"SIZE_X": 500.0, "SIZE_Y": 500.0},
            "BLOCK2": {"SIZE_X": 500.0, "SIZE_Y": 500.0}
        }
    }

    with open('test_tvc_blockage.json', 'w') as f:
        json.dump(tvc_data, f, indent=2)

    # I/O 檔案
    with open('test_io.txt', 'w') as f:
        f.write("PAD_IN {0 500 30 500 30 530 0 530}\n")
        f.write("PAD_OUT {2970 500 3000 500 3000 530 2970 530}\n")
        f.write("PAD_TOP {1000 1970 1030 1970 1030 2000 1000 2000}\n")
        f.write("PAD_BOT {1000 0 1030 0 1030 30 1000 30}\n")

    # 約束檔案 (pipe 類型不需要 Element Type 行)
    with open('test_constraints.phy', 'w') as f:
        f.write("Preplace Type: close to target\n")
        f.write("Target Type: pin\n") # 使用 pin target
        f.write("Target: u_block1\n") # 會靠近u_block1的pin box中心
        f.write("Element: u_test/inst1\n")
        f.write("Element Type: single\n")
        f.write("Area: 20.0\n\n")

        f.write("Preplace Type: close to target\n")
        f.write("Target Type: cell\n") # 使用 cell target
        f.write("Target: PAD_OUT\n") # 會靠近PAD_OUT的中心
        f.write("Element: u_test/inst2\n")
        f.write("Element Type: single\n")
        f.write("Area: 20.0\n\n")

        f.write("Preplace Type: close to target\n")
        f.write("Target Type: coords\n") # 使用 coords target
        f.write("Target: [150.0, 150.0]\n") # 靠近 Core Area 左下角 (會被 Core Edge 擋住，會往外找)
        f.write("Element: u_test/inst3\n")
        f.write("Element Type: single\n")
        f.write("Area: 20.0\n\n")

        f.write("Preplace Type: close to target\n")
        f.write("Target Type: coords\n")
        f.write("Target: [1500.0, 1000.0]\n") # 靠近中間的 blockage，會避開
        f.write("Element: u_test/inst4\n")
        f.write("Element Type: single\n")
        f.write("Area: 20.0\n\n")

        f.write("Preplace Type: close to target\n")
        f.write("Target Type: cell\n")
        f.write("Target: u_block2\n") # 靠近u_block2，但會避開u_block2
        f.write("Element: u_test/inst5\n")
        f.write("Element Type: single\n")
        f.write("Area: 20.0\n\n")

        f.write("Preplace Type: close to target\n")
        f.write("Target Type: coords\n")
        f.write("Target: [1300.0, 1300.0]\n") # 中間偏左，分配群組
        f.write("Element: GROUP_A/reg_0 GROUP_A/reg_1 GROUP_A/reg_2\n")
        f.write("Element Type: instancesgroup\n")
        f.write("Area: 50000.0\n\n") # 大一些的群組

        f.write("Preplace Type: pipe\n")  # pipe 類型
        f.write("Start: PAD_BOT\n")
        f.write("End: PAD_TOP\n")
        f.write("Stage1: u_dft/s_0 u_dft/s_1 u_dft/s_2\n")
        f.write("Stage2: u_dft/s_3 u_dft/s_4 u_dft/s_5\n")
        # 不需要 Element Type 行，因為 pipe 固定為 single
        f.write("Area: 20.0\n\n")

        f.write("Preplace Type: pipe\n")  # pipe 類型
        f.write("Start: u_test/inst1\n") # 以已放置的元件為起點
        f.write("End: u_test/inst5\n")   # 以已放置的元件為終點
        f.write("Stage1: u_dft_sub/s_0 u_dft_sub/s_1 u_dft_sub/s_2\n") # 短一些的 DFT 鏈
        # 不需要 Element Type 行
        f.write("Area: 20.0\n\n")

        f.write("Preplace Type: close to target\n")
        f.write("Target Type: coords\n")
        f.write("Target: [100.0, 100.0]\n") # 在 Core Area 之外, 無法放置
        f.write("Element: u_test/inst_fail\n")
        f.write("Element Type: single\n")
        f.write("Area: 20.0\n\n")

    print("Test files created with blockages and updated pipe constraints!")


# 使用範例
if __name__ == "__main__":
    import sys

    # 預設使用新的根模組名稱
    ROOT_MODULE_NAME = "SoIC_A16_eTV5_root"

    if len(sys.argv) == 3:
        # 如果從命令行傳入參數，可以考慮讓用戶指定 root_module_name
        # 這裡簡化為直接使用預設值
        placer = GridBasedPlacer(grid_size=50.0, root_module_name=ROOT_MODULE_NAME)
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
            constraints='test_constraints.phy',
            debug_plot_interval=0.5 # 設定每次更新的間隔時間 (秒)
        )
