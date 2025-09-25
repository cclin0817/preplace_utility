
"""
Placement and allocation algorithms for Grid-Based Pre-placement Tool
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List

from data_structures import Point, Rectangle, Block, Constraint


class PlacementEngine:
    """Mixin class for placement and allocation operations"""

    def get_target_point_for_constraint(self, constraint: Constraint) -> Optional[Point]:
        """Parse actual coordinates of target point based on constraint"""
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
            # For cell type, check if pin_box exists
            if constraint.target_name in self.blocks:
                block = self.blocks[constraint.target_name]
                # Prefer pin_box, fallback to boundary center
                if block.pin_box:
                    return block.pin_box.center()
                else:
                    return block.boundary.center()
            elif constraint.target_name in self.io_pads:
                io_pad = self.io_pads[constraint.target_name]
                if io_pad.pin_box:
                    return io_pad.pin_box.center()
                else:
                    return io_pad.boundary.center()
            elif constraint.target_name in self.mp_sensors:
                mp_sensor = self.mp_sensors[constraint.target_name]
                if mp_sensor.pin_box:
                    return mp_sensor.pin_box.center()
                else:
                    return mp_sensor.boundary.center()
        elif constraint.target_type == 'sram':
            # For SRAM, we don't return a single target point
            # The processing is handled differently in process_constraints
            return None
        elif constraint.target_type == 'sram_group':
            # sram_group doesn't have a single target point
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
        """Allocate rectangular region for instance group"""
        grids_needed = int(np.ceil(area / (self.grid_size * self.grid_size)))
        side_grids_min = int(np.ceil(np.sqrt(grids_needed)))

        target_grid_x, target_grid_y = near_point.to_grid(self.grid_size)

        best_region_rect = None
        min_dist_sq = float('inf')

        # Limit search range to avoid infinite expansion
        search_limit = max(self.grid_width, self.grid_height) // 2

        # Spiral search outward from target point
        for radius in range(search_limit + 1):
            # Traverse square boundary centered at (target_grid_x, target_grid_y) with radius
            current_points_to_check = set()
            if radius == 0:
                current_points_to_check.add((target_grid_x, target_grid_y))
            else:
                for dx in range(-radius, radius + 1):
                    # Check boundary points
                    if 0 <= target_grid_y - radius < self.grid_height:  # Bottom edge
                        current_points_to_check.add((target_grid_x + dx, target_grid_y - radius))
                    if 0 <= target_grid_y + radius < self.grid_height:  # Top edge
                        current_points_to_check.add((target_grid_x + dx, target_grid_y + radius))
                for dy in range(-radius + 1, radius):  # Left/right edges, avoid duplicate corners
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
                            is_free = False  # Out of grid range
                            break
                        if self.grid_map[y_grid, x_grid] != self.GRID_FREE:
                            is_free = False
                            break
                    if not is_free:
                        break

                if is_free:
                    # Calculate distance from region center to near_point
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

            if best_region_rect:  # If found at current radius, stop searching
                break

        if best_region_rect:
            # Mark region as reserved
            llx_grid, lly_grid, urx_grid, ury_grid = best_region_rect.to_grid_region(self.grid_size)

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
        First check instance_locations, then check blocks, IO pads, and MP sensors.
        """
        # Check in instance_locations first
        if inst_name in self.instance_locations:
            return self.instance_locations[inst_name]

        # Check in blocks
        if inst_name in self.blocks:
            return self.blocks[inst_name].boundary.center()

        # Check in IO pads
        if inst_name in self.io_pads:
            return self.io_pads[inst_name].boundary.center()

        # Check in MP sensors
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
                    # Handle SRAM_GROUP target type
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

                            # Check if area is sufficient
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

                            # Record each instance in the group at the center
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
                                'is_sram': True  # Mark as SRAM group
                            }

                            # Record each instance in the group at the center
                            center_point = region.center()
                            for element in constraint.elements:
                                self.instance_locations[element] = center_point
                                print(f"  → Placed {element} in SRAM group {group_name} at center {center_point}")

                            # For visualization, set target point at pin center
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

                        # Process module and instancesgroup the same
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

                                # Record each instance in the group at the center
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

                    # Convert start/end positions to grid coordinates
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
                                    'is_pipe_stage': True  # Mark as pipe stage
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

