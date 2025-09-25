"""
File loading and parsing functionality for Grid-Based Pre-placement Tool
"""

import json
import re
from typing import List

from data_structures import (
    Point, Rectangle, IPDefinition, Block, Blockage,
    IOPad, MPSensor, Constraint
)


class FileLoader:
    """Mixin class for file loading and parsing operations"""

    def find_block_by_design_name(self, block_shapes, design_name):
        """
        Recursively find the first block with matching DesignName in the hierarchy.

        Args:
            block_shapes: List of block shape dictionaries
            design_name: Target DesignName to find

        Returns:
            The first matching block dictionary or None
        """
        for block in block_shapes:
            if block.get('DesignName') == design_name:
                return block

            # Recursively search in SubBlockShapes
            if 'SubBlockShapes' in block:
                result = self.find_block_by_design_name(block['SubBlockShapes'], design_name)
                if result:
                    return result

        return None

    def translate_coordinates(self, coords, offset_x, offset_y):
        """
        Translate coordinates by given offset.

        Args:
            coords: List of [x, y] coordinates
            offset_x: X offset to subtract
            offset_y: Y offset to subtract

        Returns:
            Translated coordinates
        """
        return [[x - offset_x, y - offset_y] for x, y in coords]

    def load_blocks_from_subblocks(self, sub_blocks, parent_offset_x=0, parent_offset_y=0):
        """
        Load blocks from SubBlockShapes with coordinate translation.

        Args:
            sub_blocks: List of SubBlockShape dictionaries
            parent_offset_x: Parent's offset in X (for relative coords)
            parent_offset_y: Parent's offset in Y (for relative coords)
        """
        for block_data in sub_blocks:
            # Calculate actual coordinates (relative to new origin)
            coords = block_data['Coords']
            # SubBlockShapes coordinates are already relative, just apply parent offset
            actual_coords = [[x + parent_offset_x, y + parent_offset_y] for x, y in coords]

            block = Block(
                name=block_data['BlockName'],
                boundary=Rectangle(
                    min(c[0] for c in actual_coords),
                    min(c[1] for c in actual_coords),
                    max(c[0] for c in actual_coords),
                    max(c[1] for c in actual_coords)
                )
            )

            # Process PinCoords if present
            if 'PinCoords' in block_data and block_data['PinCoords']:
                pin_list = block_data['PinCoords']
                if pin_list and pin_list[0]:
                    first_pin_coords = pin_list[0]
                    # Apply same offset to pin coordinates
                    actual_pin_coords = [[x + parent_offset_x, y + parent_offset_y]
                                       for x, y in first_pin_coords]
                    block.pin_box = Rectangle(
                        min(c[0] for c in actual_pin_coords),
                        min(c[1] for c in actual_pin_coords),
                        max(c[0] for c in actual_pin_coords),
                        max(c[1] for c in actual_pin_coords)
                    )

            self.blocks[block.name] = block

    def load_blockages_from_block(self, block_data, offset_x, offset_y):
        """
        Load blockages from a block's Blockages field.

        Args:
            block_data: Block dictionary potentially containing Blockages
            offset_x: X offset for coordinate translation
            offset_y: Y offset for coordinate translation
        """
        if 'Blockages' in block_data:
            blockages_data = block_data['Blockages']
            for idx, blockage_coords_list in enumerate(blockages_data):
                coords = blockage_coords_list
                if coords and coords[-1] == coords[0]:
                    coords = coords[:-1]  # Remove duplicate last point

                if not coords:
                    continue

                # Apply offset to blockage coordinates
                translated_coords = self.translate_coordinates(coords, offset_x, offset_y)

                x_coords = [c[0] for c in translated_coords]
                y_coords = [c[1] for c in translated_coords]

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

    def load_tvc_json(self, filepath: str):
        """Load TVC JSON file - Modified to support any DesignName as root module"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Load IP definitions first (always needed)
        if 'IPS' in data:
            for ip_name, ip_data in data['IPS'].items():
                if 'SIZE_X' in ip_data and 'SIZE_Y' in ip_data:
                    self.ip_definitions[ip_name] = IPDefinition(
                        name=ip_name,
                        size_x=float(ip_data['SIZE_X']),
                        size_y=float(ip_data['SIZE_Y'])
                    )
            print(f"    ✓ Loaded {len(self.ip_definitions)} IP definitions.")

        # Check if root_module_name is a top-level key
        if 'S1_output' in data and self.root_module_name in data['S1_output']:
            # Original logic for top-level module
            print(f"    ✓ Using top-level module: {self.root_module_name}")

            tvc_root = data['S1_output'][self.root_module_name]
            top_shape = tvc_root['TOPShape']

            # Die boundary
            die_coords = top_shape['DieCoords']
            self.die_boundary = Rectangle(
                die_coords[0][0], die_coords[0][1],
                die_coords[2][0], die_coords[2][1]
            )

            # Core area
            core_coords = top_shape['CoreCoords']
            self.core_area = Rectangle(
                core_coords[0][0], core_coords[0][1],
                core_coords[2][0], core_coords[2][1]
            )

            # Blocks
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

                    # Pin box processing
                    if 'PinCoords' in block_data and block_data['PinCoords']:
                        pin_list = block_data['PinCoords']
                        if pin_list and pin_list[0]:
                            first_pin_coords = pin_list[0]
                            block.pin_box = Rectangle(
                                min(c[0] for c in first_pin_coords),
                                min(c[1] for c in first_pin_coords),
                                max(c[0] for c in first_pin_coords),
                                max(c[1] for c in first_pin_coords)
                            )
                    self.blocks[block.name] = block

            # Load blockages from TOPShape
            if 'Blockages' in top_shape:
                blockages_data = top_shape['Blockages']
                for idx, blockage_coords_list in enumerate(blockages_data):
                    coords = blockage_coords_list
                    if coords and coords[-1] == coords[0]:
                        coords = coords[:-1]

                    if not coords:
                        continue

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

            # Load MP Sensors
            if 'Solution_MP_SENSOR' in data and self.root_module_name in data['Solution_MP_SENSOR']:
                if 'MP_SENSOR' in data['Solution_MP_SENSOR'][self.root_module_name]:
                    mp_sensors = data['Solution_MP_SENSOR'][self.root_module_name]['MP_SENSOR']
                    for sensor_data in mp_sensors:
                        name = sensor_data['Name']
                        cell_name = sensor_data['CellName']
                        location = sensor_data['Location']

                        if cell_name in self.ip_definitions:
                            size_x, size_y = self.ip_definitions[cell_name].get_size()
                            llx, lly = location[0], location[1]
                            urx, ury = llx + size_x, lly + size_y

                            self.mp_sensors[name] = MPSensor(
                                name=name,
                                boundary=Rectangle(llx, lly, urx, ury),
                                pin_box=None
                            )
                        else:
                            print(f"警告: MP_SENSOR '{name}' 的 CellName '{cell_name}' 在 IP 尺寸中未找到，跳過。")

            # Load I/O Pads
            if 'Solution_GPIO' in data and self.root_module_name in data['Solution_GPIO']:
                if 'GPIO' in data['Solution_GPIO'][self.root_module_name]:
                    gpio_pads = data['Solution_GPIO'][self.root_module_name]['GPIO']
                    for pad_data in gpio_pads:
                        name = pad_data['Name']
                        cell_name = pad_data['CellName']
                        location = pad_data['Location']

                        if cell_name in self.ip_definitions:
                            size_x, size_y = self.ip_definitions[cell_name].get_size()
                            llx, lly = location[0], location[1]
                            urx, ury = llx + size_x, lly + size_y

                            self.io_pads[name] = IOPad(
                                name=name,
                                boundary=Rectangle(llx, lly, urx, ury),
                                pin_box=None
                            )
                        else:
                            print(f"警告: I/O Pad '{name}' 的 CellName '{cell_name}' 在 IP 尺寸中未找到，跳過。")

        else:
            # New logic: Search for DesignName in the hierarchy
            print(f"    ✓ Searching for DesignName: {self.root_module_name} in hierarchy...")

            # Get all top-level blocks
            found_block = None
            for top_key in data.get('S1_output', {}).keys():
                tvc_root = data['S1_output'][top_key]
                if 'BlockShapes' in tvc_root:
                    found_block = self.find_block_by_design_name(
                        tvc_root['BlockShapes'],
                        self.root_module_name
                    )
                    if found_block:
                        print(f"    ✓ Found DesignName '{self.root_module_name}' as '{found_block['BlockName']}'")
                        break

            if not found_block:
                raise ValueError(f"Cannot find DesignName '{self.root_module_name}' in the hierarchy")

            # Use the found block as root module
            # Calculate offset to make lower-left corner [0, 0]
            coords = found_block['Coords']
            offset_x = min(c[0] for c in coords)
            offset_y = min(c[1] for c in coords)

            # Set die and core boundaries based on the block
            translated_coords = self.translate_coordinates(coords, offset_x, offset_y)
            self.die_boundary = Rectangle(
                min(c[0] for c in translated_coords),
                min(c[1] for c in translated_coords),
                max(c[0] for c in translated_coords),
                max(c[1] for c in translated_coords)
            )
            self.core_area = Rectangle(
                self.die_boundary.llx,
                self.die_boundary.lly,
                self.die_boundary.urx,
                self.die_boundary.ury
            )

            # Load SubBlockShapes as blocks
            if 'SubBlockShapes' in found_block:
                print(f"    ✓ Loading {len(found_block['SubBlockShapes'])} sub-blocks")
                # SubBlockShapes coordinates are relative, but we already offset to [0,0]
                # So we pass negative offset to maintain relative positions
                self.load_blocks_from_subblocks(found_block['SubBlockShapes'], 0, 0)

            # Load Blockages from the block (if any)
            self.load_blockages_from_block(found_block, offset_x, offset_y)

            # No I/O pads or MP sensors for non-top-level modules
            print(f"    ✓ Note: No I/O pads or MP sensors for non-top-level module")

    def load_constraints(self, filepath: str):
        """Load constraints file"""
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
                    constraint.element_type = 'instancesgroup'  # Default for pipe
                    constraint.stages = []  # Initialize stages list

                i += 1
                while i < len(lines):
                    line = lines[i].strip()
                    if not line:  # Empty line ends a constraint
                        break
                    if line.startswith('Preplace Type:'):  # Next constraint
                        i -= 1  # Back up one line for outer loop to process
                        break

                    # close_to_target related parsing
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
                            elif constraint.target_type == 'sram_group':  # Parse multiple blocks
                                constraint.target_blocks = target_value.split()
                                constraint.target_name = target_value  # Keep original string
                            else:  # 'cell' or 'pin' or 'sram'
                                constraint.target_name = target_value
                        elif line.startswith('Element Type:'):
                            constraint.element_type = line.split(':')[1].strip()
                        elif line.startswith('Area:'):
                            constraint.area = float(line.split(':')[1].strip())
                        elif line.startswith('Element:'):
                            constraint.elements = line.split(':')[1].strip().split()

                    # pipe related parsing
                    elif constraint.type == 'pipe':
                        if line.startswith('Start:'):
                            constraint.start = line.split(':')[1].strip()
                        elif line.startswith('End:'):
                            constraint.end = line.split(':')[1].strip()
                        elif line.startswith('Stage'):  # Stage1, Stage2, ...
                            stage_elements = line.split(':')[1].strip().split()
                            if stage_elements:  # Only add non-empty stages
                                constraint.stages.append(stage_elements)  # Add as a stage
                                constraint.elements.extend(stage_elements)  # Keep for compatibility

                    i += 1

                # Only add constraint if it has elements (or stages for pipe)
                if constraint.type == 'pipe' and constraint.stages:
                    self.constraints.append(constraint)
                elif constraint.type == 'close_to_target' and constraint.elements:
                    self.constraints.append(constraint)

            i += 1

