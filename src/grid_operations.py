"""
Grid building and manipulation operations for Grid-Based Pre-placement Tool
"""

import numpy as np
import heapq
from typing import Tuple, Optional, List

from data_structures import Point, Rectangle


class GridOperations:
    """Mixin class for grid operations"""

    def is_valid_grid_coord(self, x: int, y: int) -> bool:
        """Unified grid coordinate validity check"""
        return 0 <= x < self.grid_width and 0 <= y < self.grid_height

    def build_grid_map(self):
        """Build grid map"""
        if not self.die_boundary or not self.core_area:
            raise ValueError("Die boundary or core area not loaded.")

        # Calculate grid size
        self.grid_width = int(np.ceil(self.die_boundary.urx / self.grid_size))
        self.grid_height = int(np.ceil(self.die_boundary.ury / self.grid_size))

        # Initialize grid (0 = available GRID_FREE)
        self.grid_map = np.zeros((self.grid_height, self.grid_width), dtype=int)

        # Mark areas outside core as unavailable (GRID_BLOCKED)
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                px = (x + 0.5) * self.grid_size
                py = (y + 0.5) * self.grid_size

                if not self.core_area.contains_point(Point(px, py)):
                    self.grid_map[y, x] = self.GRID_BLOCKED

        # Mark blockages as occupied (GRID_BLOCKED) - before blocks
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

        # Mark blocks as occupied (GRID_BLOCKED)
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

        # Mark MP sensors as occupied
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

    def find_nearest_free_grid(self, target: Point) -> Optional[Tuple[int, int]]:
        """
        Find nearest available grid (GRID_FREE).
        Uses BFS for search.
        """
        target_grid_x, target_grid_y = target.to_grid(self.grid_size)

        # Adjust target grid to valid range
        target_grid_x = max(0, min(target_grid_x, self.grid_width - 1))
        target_grid_y = max(0, min(target_grid_y, self.grid_height - 1))

        # If target point itself is available
        if self.grid_map[target_grid_y, target_grid_x] == self.GRID_FREE:
            return (target_grid_x, target_grid_y)

        # BFS search for nearest available grid
        visited = set()
        queue = [(target_grid_x, target_grid_y)]

        while queue:
            x, y = queue.pop(0)

            if (x, y) in visited:
                continue
            visited.add((x, y))

            if self.grid_map[y, x] == self.GRID_FREE:
                return (x, y)

            # Explore four directions (expanding from center)
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if self.is_valid_grid_coord(nx, ny) and (nx, ny) not in visited:
                    queue.append((nx, ny))

        return None  # No available grid found

    def a_star_path(self, start: Tuple[int, int],
                    end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Use A* algorithm to find path, only avoiding GRID_BLOCKED.
        Allows passing through GRID_FREE and GRID_RESERVED areas.

        Returns:
            List of grid coordinates, or None if no path found.
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

            # Explore neighbors
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                nx, ny = neighbor

                if (self.is_valid_grid_coord(nx, ny) and
                    self.grid_map[ny, nx] != self.GRID_BLOCKED and  # Can go through FREE or RESERVED
                    neighbor not in visited_nodes):

                    tentative_g_score = g_score + 1

                    if neighbor not in g_scores or tentative_g_score < g_scores[neighbor]:
                        g_scores[neighbor] = tentative_g_score
                        h_score = heuristic(neighbor, end)
                        f_score = tentative_g_score + h_score
                        heapq.heappush(open_set, (f_score, tentative_g_score, neighbor, path + [neighbor]))

        return None  # No path found

    def find_max_rectangle_in_region(self, bounding_rect: Rectangle) -> Optional[Rectangle]:
        """
        Find the largest rectangle of FREE grids within the given bounding rectangle.
        Uses a corrected histogram-based algorithm.
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

        # Use histogram-based maximal rectangle algorithm
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

        # For each row, find the maximum rectangle using corrected histogram approach
        for i in range(rows):
            # For each column as a potential starting point
            for start_col in range(cols):
                if heights[i, start_col] == 0:
                    continue

                # Try different heights starting from this column
                min_height = heights[i, start_col]

                # Extend to the right and track minimum height
                for end_col in range(start_col, cols):
                    if heights[i, end_col] == 0:
                        break

                    # Update minimum height for this range
                    min_height = min(min_height, heights[i, end_col])

                    # Calculate area for this rectangle
                    width = end_col - start_col + 1
                    area = width * min_height

                    # Calculate the actual rectangle coordinates
                    rect_grid_llx = grid_llx + start_col
                    rect_grid_lly = grid_lly + i - min_height + 1
                    rect_grid_urx = grid_llx + end_col + 1
                    rect_grid_ury = grid_lly + i + 1

                    # Verify that all cells in this rectangle are FREE
                    # (This is a safety check - should be guaranteed by the algorithm)
                    all_free = True
                    for check_y in range(rect_grid_lly - grid_llx, rect_grid_ury - grid_llx):
                        for check_x in range(start_col, end_col + 1):
                            if check_y < 0 or check_y >= rows or sub_grid[check_y, check_x] != self.GRID_FREE:
                                all_free = False
                                break
                        if not all_free:
                            break

                    if not all_free:
                        continue

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
