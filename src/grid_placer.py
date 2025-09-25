"""
Grid-Based Pre-placement Tool - Main Placer Class
Coordinates all placement operations and manages the overall flow
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from data_structures import (
    Point, Rectangle, IPDefinition, Block, Blockage,
    IOPad, MPSensor, Constraint
)
from file_loaders import FileLoader
from grid_operations import GridOperations
from placement_engine import PlacementEngine
from visualization import Visualization
from output_generator import OutputGenerator


class GridBasedPlacer(FileLoader, GridOperations, PlacementEngine,
                      Visualization, OutputGenerator):
    """Main class for grid-based placement using multiple inheritance for functionality"""

    def __init__(self, grid_size: float = 50.0, root_module_name: str = "uDue1/u_socss_0"):
        """
        Initialize the grid-based placer

        Args:
            grid_size: Grid size in micrometers
            root_module_name: Name of the root module in TVC JSON
        """
        self.grid_size = grid_size
        self.root_module_name = root_module_name

        # Design information
        self.die_boundary: Optional[Rectangle] = None
        self.core_area: Optional[Rectangle] = None
        self.blocks: Dict[str, Block] = {}
        self.blockages: Dict[str, Blockage] = {}
        self.io_pads: Dict[str, IOPad] = {}
        self.mp_sensors: Dict[str, MPSensor] = {}
        self.ip_definitions: Dict[str, IPDefinition] = {}

        # Grid map
        self.grid_map: Optional[np.ndarray] = None
        self.grid_width: int = 0
        self.grid_height: int = 0

        # Constraints and placement results
        self.constraints: List[Constraint] = []
        self.placements: Dict[str, any] = {}
        self.instance_locations: Dict[str, Point] = {}
        self.failed_constraints: List[str] = []

        # Grid state definitions
        self.GRID_FREE = 0      # Available
        self.GRID_BLOCKED = 1   # Blocked by hard blocks/blockages/core boundary
        self.GRID_RESERVED = 2  # Reserved for instance groups

        # Visualization related member variables
        self.fig = None
        self.ax_grid = None
        self.ax_placement = None
        self.grid_img = None
        self.placement_artists = []
        self.current_constraint_text = None
        self.cmap_constraints = None

    def run(self, tvc_json: str, constraints: str,
            output_tcl: str = 'dft_regs_pre_place.tcl',
            failed_list: str = 'failed_preplace.list'):
        """
        Execute the complete placement flow

        Args:
            tvc_json: Path to TVC JSON file
            constraints: Path to constraints file
            output_tcl: Output TCL script filename
            failed_list: Output failed list filename
        """
        print("=" * 60)
        print("Grid-Based Pre-placement Tool with Blockage Support")
        print(f"Grid Size: {self.grid_size} μm")
        print(f"Root Module Name: {self.root_module_name}")
        print("=" * 60)

        # 1. Load files
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
        sram_count = sum(1 for c in self.constraints
                         if c.type == 'close_to_target' and c.target_type == 'sram')
        sram_group_count = sum(1 for c in self.constraints
                              if c.type == 'close_to_target' and c.target_type == 'sram_group')

        print(f"    ✓ Constraint types: {close_count} close_to_target "
              f"({sram_count} SRAM, {sram_group_count} SRAM_GROUP), {pipe_count} pipe")
        print(f"    ✓ Processing order: close_to_target first, then pipe")
        print(f"    ✓ Pipe constraints will create stage groups (one grid per stage)")

        if sram_count > 0:
            print(f"    ✓ SRAM constraints will place groups adjacent to target block pin edges")
        if sram_group_count > 0:
            print(f"    ✓ SRAM_GROUP constraints will find max rectangle within "
                  f"bounding box of multiple blocks")

        # Visualize initial design layout (Step 1)
        print("\n[VISUALIZATION] Step 1: Showing initial design layout...")
        self.visualize_initial_design()

        # 2. Build grid
        print("\n[2] Building grid map...")
        self.build_grid_map()
        print(f"    ✓ Grid dimensions: {self.grid_width} x {self.grid_height} grids")

        # Statistics for grid state
        free_grids = np.sum(self.grid_map == self.GRID_FREE)
        blocked_grids = np.sum(self.grid_map == self.GRID_BLOCKED)
        total_grids = self.grid_width * self.grid_height

        print(f"    ✓ Initial Free grids: {free_grids}/{total_grids} "
              f"({100*free_grids/total_grids:.1f}%)")
        print(f"    ✓ Initial Blocked grids: {blocked_grids} grids")

        # Visualize grid state (Step 2)
        print("\n[VISUALIZATION] Step 2: Showing grid state map...")
        self.visualize_grid_state()

        # 3. Process constraints
        print("\n[3] Processing constraints with live visualization...")
        print("    (Step 3: Dynamic constraint processing visualization)")

        # Setup dynamic visualization
        self._setup_placement_visualization()
        self.process_constraints()

        # After processing all constraints, keep figure displayed
        print("\n[VISUALIZATION] Final placement results")

        # Remove last constraint label
        if self.current_constraint_text:
            self.current_constraint_text.remove()
            self.current_constraint_text = None

        self.fig.suptitle('Final Placement Results', fontsize=16, fontweight='bold')
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        print("\n>>> Please close the figure window to generate output files...")
        plt.show()  # Blocking mode for user to check final results

        # 4. Generate outputs
        print("\n[4] Generating outputs...")
        self.generate_tcl_script(output_tcl)
        print(f"    ✓ TCL script: {output_tcl}")

        self.save_failed_list(failed_list)
        print(f"    ✓ Failed list: {failed_list}")

        # Statistics for final results
        print("\n[5] Summary:")
        successful_placements = len(self.placements)
        total_instances_placed = len(self.instance_locations)

        print(f"    ✓ Successfully placed: {successful_placements} items (groups and instances)")
        print(f"    ✓ Total individual instances placed: {total_instances_placed}")
        print(f"    ✓ Failed constraints: {len(self.failed_constraints)}")

        print("\n" + "=" * 60)
        print("✓ Complete! All processing finished.")
        print("=" * 60)

