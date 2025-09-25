"""
Data structures for Grid-Based Pre-placement Tool
Contains all dataclasses and their methods
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Point:
    """Coordinate point"""
    x: float
    y: float

    def to_grid(self, grid_size: float) -> Tuple[int, int]:
        """Convert to grid coordinates"""
        return (int(self.x / grid_size), int(self.y / grid_size))

    def __str__(self):
        return f"({self.x:.2f}, {self.y:.2f})"


@dataclass
class Rectangle:
    """Rectangular region"""
    llx: float  # Lower left x
    lly: float  # Lower left y
    urx: float  # Upper right x
    ury: float  # Upper right y

    def contains_point(self, point: Point) -> bool:
        """Check if point is within rectangle"""
        return (self.llx <= point.x <= self.urx and
                self.lly <= point.y <= self.ury)

    def to_grid_region(self, grid_size: float) -> Tuple[int, int, int, int]:
        """Convert to grid region (llx, lly, urx, ury grid indices)"""
        return (
            int(self.llx / grid_size),
            int(self.lly / grid_size),
            int(np.ceil(self.urx / grid_size)),
            int(np.ceil(self.ury / grid_size))
        )

    def center(self) -> Point:
        """Return center point of rectangle"""
        return Point((self.llx + self.urx) / 2, (self.lly + self.ury) / 2)


@dataclass
class IPDefinition:
    """IP definition data structure"""
    name: str
    size_x: float
    size_y: float

    def get_size(self) -> Tuple[float, float]:
        return (self.size_x, self.size_y)


@dataclass
class Block:
    """Hard block in the design"""
    name: str
    boundary: Rectangle
    pin_box: Optional[Rectangle] = None


@dataclass
class Blockage:
    """Blockage region (treated as GRID_BLOCKED)"""
    name: str
    boundary: Rectangle


@dataclass
class IOPad:
    """I/O Pad data structure"""
    name: str
    boundary: Rectangle
    pin_box: Optional[Rectangle] = None


@dataclass
class MPSensor:
    """MP Sensor data structure"""
    name: str
    boundary: Rectangle
    pin_box: Optional[Rectangle] = None


@dataclass
class Constraint:
    """Placement constraint"""
    type: str  # 'close_to_target', 'pipe'
    elements: List[str] = field(default_factory=list)  # Affected elements

    # close_to_target related
    target_type: Optional[str] = None  # 'cell', 'coords', 'pin', 'sram', 'sram_group'
    target_name: Optional[str] = None  # Target name (cell) or space-separated list for sram_group
    target_coords: Optional[Point] = None  # Target coordinates (coords)
    target_blocks: Optional[List[str]] = None  # For sram_group, list of block names

    # pipe related
    start: Optional[str] = None  # Pipe start element
    end: Optional[str] = None  # Pipe end element
    stages: Optional[List[List[str]]] = None  # List of stages, each stage is a list of instances

    element_type: str = 'single'  # 'single', 'instancesgroup', 'module'
    area: float = 20.0  # Only for instancesgroup/module

    # Visualization helpers
    vis_target_point: Optional[Point] = None  # close_to_target actual resolved target point
    vis_start_point: Optional[Point] = None  # pipe actual resolved start point
    vis_end_point: Optional[Point] = None  # pipe actual resolved end point
    vis_pipe_path: Optional[List[Tuple[int, int]]] = None  # pipe actual path used
    vis_bounding_rect: Optional[Rectangle] = None  # For sram_group bounding rectangle

    def __str__(self):
        if self.type == 'close_to_target':
            target_info = ""
            if self.target_type == 'coords' and self.target_coords:
                target_info = f"coords={self.target_coords}"
            elif self.target_type in ['cell', 'pin', 'sram'] and self.target_name:
                target_info = f"{self.target_type}='{self.target_name}'"
            elif self.target_type == 'sram_group' and self.target_blocks:
                target_info = f"sram_group={self.target_blocks}"
            return f"Constraint(Type='{self.type}', Target={target_info})"
        elif self.type == 'pipe':
            num_stages = len(self.stages) if self.stages else 0
            return f"Constraint(Type='{self.type}', Start='{self.start}', End='{self.end}', Stages={num_stages})"
        else:
            return super().__str__()  # For unknown types, use default dataclass string representation

