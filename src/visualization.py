"""
Visualization functionality for Grid-Based Pre-placement Tool
Handles all plotting and visual representation of the placement process
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional

from data_structures import Point, Rectangle

# Check matplotlib version for correct API usage
MATPLOTLIB_VERSION = tuple(map(int, matplotlib.__version__.split('.')[:2]))
USE_NEW_CMAP_API = MATPLOTLIB_VERSION >= (3, 7)


class Visualization:
    """Mixin class for visualization operations"""

    def _draw_static_design_elements(self, ax, show_legend=True):
        """
        Helper function: Draw static design elements (chip boundary, core area, blocks, blockages, I/O)

        Args:
            ax: Matplotlib axis to draw on
            show_legend: Whether to show legend
        """
        # Chip boundary
        if self.die_boundary:
            die_rect = patches.Rectangle(
                (self.die_boundary.llx, self.die_boundary.lly),
                self.die_boundary.urx - self.die_boundary.llx,
                self.die_boundary.ury - self.die_boundary.lly,
                linewidth=2, edgecolor='black', facecolor='none'
            )
            ax.add_patch(die_rect)

        # Core area
        if self.core_area:
            core_rect = patches.Rectangle(
                (self.core_area.llx, self.core_area.lly),
                self.core_area.urx - self.core_area.llx,
                self.core_area.ury - self.core_area.lly,
                linewidth=1.5, edgecolor='blue', facecolor='lightgray', alpha=0.3
            )
            ax.add_patch(core_rect)

        # Blockages (dark red hatched fill)
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

        # Blocks
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

            # Draw Pin Box (dark gray)
            if block.pin_box:
                pin_box_rect = patches.Rectangle(
                    (block.pin_box.llx, block.pin_box.lly),
                    block.pin_box.urx - block.pin_box.llx,
                    block.pin_box.ury - block.pin_box.lly,
                    linewidth=1, edgecolor='dimgray', facecolor='dimgray', alpha=0.6,
                    linestyle='--'  # Dashed line for easier distinction
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

        # MP Sensors
        for name, sensor in self.mp_sensors.items():
            sensor_rect = patches.Rectangle(
                (sensor.boundary.llx, sensor.boundary.lly),
                sensor.boundary.urx - sensor.boundary.llx,
                sensor.boundary.ury - sensor.boundary.lly,
                linewidth=1, edgecolor='orange', facecolor='lightyellow', alpha=0.6
            )
            ax.add_patch(sensor_rect)

        # Set axis limits and properties
        if self.die_boundary:
            ax.set_xlim(self.die_boundary.llx, self.die_boundary.urx)
            ax.set_ylim(self.die_boundary.lly, self.die_boundary.ury)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Add legend if requested
        if show_legend:
            legend_elements = [
                patches.Patch(color='wheat', label='Hard Blocks'),
                patches.Patch(color='darkred', alpha=0.5, hatch='//', label='Blockages'),
                patches.Patch(color='lightgreen', alpha=0.6, label='I/O Pads'),
                patches.Patch(color='lightyellow', alpha=0.6, label='MP Sensors'),
                patches.Patch(color='lightgray', alpha=0.3, label='Core Area')
            ]
            # Place legend outside the plot on the right side
            ax.legend(handles=legend_elements, loc='center left',
                     bbox_to_anchor=(1.02, 0.5), fontsize=9)

    def visualize_initial_design(self):
        """Visualize initial design layout without grid and placement results"""
        if not self.die_boundary:
            print("警告: 晶片邊界未載入，無法視覺化原始設計。")
            return

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        self._draw_static_design_elements(ax)
        ax.set_title('Step 1: Initial Design Layout (After Loading Files)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X (μm)', fontsize=12)
        ax.set_ylabel('Y (μm)', fontsize=12)

        # Add information text - placed at bottom left with smaller font and semi-transparent background
        info_text = f"Die: {self.die_boundary.urx:.0f}x{self.die_boundary.ury:.0f} μm | "
        info_text += f"Core: ({self.core_area.llx:.0f},{self.core_area.lly:.0f})-"
        info_text += f"({self.core_area.urx:.0f},{self.core_area.ury:.0f}) | "
        info_text += f"Blocks: {len(self.blocks)} | Blockages: {len(self.blockages)} | "
        info_text += f"I/O Pads: {len(self.io_pads)} | MP Sensors: {len(self.mp_sensors)}"

        ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray'))

        plt.tight_layout()
        print("\n>>> Please close the figure window to continue to the next step...")
        plt.show()  # Blocking mode, wait for user to close

    def visualize_grid_state(self):
        """Visualize grid state map with optimized tick display"""
        if self.grid_map is None:
            print("警告: 網格地圖未建立，無法視覺化網格狀態。")
            return

        fig, ax = plt.subplots(1, 1, figsize=(16, 10))

        # Define color mapping
        colors = ['#90EE90', '#FF6B6B', '#FFE66D']  # Green(FREE), Red(BLOCKED), Yellow(RESERVED)
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        bounds = [0, 1, 2, 3]
        norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        # Display grid map
        im = ax.imshow(self.grid_map, cmap=cmap, norm=norm, origin='lower',
                      extent=[0, self.grid_width * self.grid_size,
                              0, self.grid_height * self.grid_size],
                      aspect='equal', interpolation='nearest')

        # Optimize tick display - reduce number of ticks
        # X-axis: determine tick interval based on width
        x_max = self.grid_width * self.grid_size
        if x_max <= 1000:
            x_tick_interval = 100  # Every 100 micrometers
        elif x_max <= 5000:
            x_tick_interval = 250  # Every 250 micrometers
        elif x_max <= 10000:
            x_tick_interval = 500  # Every 500 micrometers
        else:
            x_tick_interval = 1000  # Every 1000 micrometers

        # Y-axis: determine tick interval based on height
        y_max = self.grid_height * self.grid_size
        if y_max <= 1000:
            y_tick_interval = 100
        elif y_max <= 5000:
            y_tick_interval = 250
        elif y_max <= 10000:
            y_tick_interval = 500
        else:
            y_tick_interval = 1000

        # Set major ticks
        x_ticks = np.arange(0, x_max + 1, x_tick_interval)
        y_ticks = np.arange(0, y_max + 1, y_tick_interval)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        # Add minor ticks (grid lines) without labels
        x_minor_ticks = np.arange(0, x_max + 1, self.grid_size)
        y_minor_ticks = np.arange(0, y_max + 1, self.grid_size)
        ax.set_xticks(x_minor_ticks, minor=True)
        ax.set_yticks(y_minor_ticks, minor=True)

        # Set grid line styles
        ax.grid(True, which='major', color='black', linewidth=0.8, alpha=0.7)
        ax.grid(True, which='minor', color='gray', linewidth=0.3, alpha=0.3)

        # Rotate x-axis labels to avoid overlap (if needed)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

        # Set title and labels
        ax.set_title(f'Step 2: Grid State Map (Grid Size: {self.grid_size} μm)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X (μm)', fontsize=12)
        ax.set_ylabel('Y (μm)', fontsize=12)

        # Add legend - place outside figure on right side
        legend_elements = [
            patches.Patch(color='#90EE90', label='FREE - Available for placement'),
            patches.Patch(color='#FF6B6B', label='BLOCKED - Occupied by blocks/blockages/core boundary'),
            patches.Patch(color='#FFE66D', label='RESERVED - Reserved for instance groups')
        ]
        ax.legend(handles=legend_elements, loc='center left',
                 bbox_to_anchor=(1.08, 0.5), fontsize=9)

        # Add statistics information
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
        plt.show()  # Blocking mode, wait for user to close

    def _setup_placement_visualization(self):
        """Initialize placement result visualization figures and subplots"""
        if self.die_boundary is None or self.grid_map is None:
            print("錯誤: 晶片邊界或網格地圖未載入，無法設置視覺化。")
            return

        self.fig, (self.ax_grid, self.ax_placement) = plt.subplots(1, 2, figsize=(28, 13))
        self.fig.suptitle('Step 3: Constraint Processing and Placement',
                         fontsize=16, fontweight='bold')

        # --- Left plot: Grid map ---
        colors = ['#E0FFE0', '#FFCCCC', '#FFFFCC']  # FREE (light green), BLOCKED (light red), RESERVED (light yellow)
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

        # Left plot legend - place below the plot outside
        legend_elements_ax1 = [
            patches.Patch(color='#E0FFE0', label='Free (0)'),
            patches.Patch(color='#FFCCCC', label='Blocked (1)'),
            patches.Patch(color='#FFFFCC', label='Reserved (2)')
        ]
        self.ax_grid.legend(handles=legend_elements_ax1, loc='upper center',
                           bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=9)

        # --- Right plot: Placement results ---
        self._draw_static_design_elements(self.ax_placement, show_legend=False)  # Draw static elements as background, no duplicate legend
        self.ax_placement.set_title('Placement Result with Constraint Visualization', fontsize=14)
        self.ax_placement.set_xlabel('X (μm)', fontsize=12)
        self.ax_placement.set_ylabel('Y (μm)', fontsize=12)

        # Set up color map for constraints
        if USE_NEW_CMAP_API:
            # New Matplotlib version (>= 3.7)
            self.cmap_constraints = plt.colormaps['hsv'].resampled(len(self.constraints) + 1)
        else:
            # Old Matplotlib version (< 3.7)
            self.cmap_constraints = plt.cm.get_cmap('hsv', len(self.constraints) + 1)

        # Setup right plot legend - place below outside, use two columns (update Pipe Start to triangle)
        legend_elements_ax2 = [
            patches.Patch(color='wheat', label='Hard Blocks'),
            patches.Patch(color='darkred', alpha=0.5, hatch='//', label='Blockages'),
            patches.Patch(color='lightgreen', alpha=0.6, label='I/O Pads'),
            patches.Patch(color='lightyellow', alpha=0.6, label='MP Sensors'),
            patches.Patch(color='lightgray', alpha=0.3, label='Core Area'),
            patches.Patch(color='gray', alpha=0.3, label='Instance Groups'),
            patches.Patch(color='purple', alpha=0.3, label='Pipe Stage Groups'),
            patches.Patch(color='cyan', alpha=0.4, label='SRAM Groups'),  # for single-block SRAM
            patches.Patch(color='magenta', alpha=0.4, label='SRAM Multi-Block Groups'),  # for multi-block SRAM
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                      markersize=8, label='Single Instances', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='x', color='gray', markersize=10, mew=2,
                      linestyle='None', label='Target Point'),
            plt.Line2D([0], [0], marker='^', color='gray', markersize=8, mew=1,
                      linestyle='None', label='Pipe Start'),  # Changed to triangle
            plt.Line2D([0], [0], marker='s', color='gray', markersize=8, mew=1,
                      linestyle='None', label='Pipe End')
        ]
        self.ax_placement.legend(handles=legend_elements_ax2, loc='upper center',
                                bbox_to_anchor=(0.5, -0.08), ncol=7, fontsize=9)

        # Adjust layout to accommodate suptitle and bottom legend
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        plt.show(block=False)  # Non-blocking mode

    def _update_placement_visualization(self, current_constraint_idx: int):
        """
        Update placement visualization

        Args:
            current_constraint_idx: Index of current constraint being processed
        """
        if self.fig is None:
            return

        # Update grid map
        self.grid_img.set_data(self.grid_map)

        # Clear old placement results (including all dynamically added elements)
        for artist in self.placement_artists:
            if hasattr(artist, 'remove'):
                artist.remove()
        self.placement_artists.clear()

        # Redraw constraint target points/start points/end points
        for idx, constraint in enumerate(self.constraints):
            color = self.cmap_constraints(idx / len(self.constraints))

            if constraint.type == 'close_to_target':
                # Draw bounding rectangle for sram_group
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
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none',
                                 boxstyle='round,pad=0.1'),
                        zorder=10
                    )
                    self.placement_artists.append(artist_text)

                if constraint.vis_target_point:
                    target_point = constraint.vis_target_point
                    artist = self.ax_placement.plot(target_point.x, target_point.y, 'x',
                                                   color=color, markersize=10, mew=2, zorder=10)[0]
                    self.placement_artists.append(artist)
                    artist_text = self.ax_placement.annotate(f'C{idx} Target',
                                                            (target_point.x, target_point.y),
                                                            fontsize=7, color=color,
                                                            xytext=(5, 5), textcoords='offset points',
                                                            bbox=dict(facecolor='white', alpha=0.7,
                                                                    edgecolor='none',
                                                                    boxstyle='round,pad=0.1'),
                                                            zorder=10)
                    self.placement_artists.append(artist_text)

            elif constraint.type == 'pipe':
                if constraint.vis_start_point:
                    artist = self.ax_placement.plot(constraint.vis_start_point.x,
                                                   constraint.vis_start_point.y, '^',
                                                   color=color, markersize=8, mew=1, zorder=10)[0]
                    self.placement_artists.append(artist)
                    artist_text = self.ax_placement.annotate(f'C{idx} Start',
                                                            (constraint.vis_start_point.x,
                                                             constraint.vis_start_point.y),
                                                            fontsize=7, color=color,
                                                            xytext=(5, 5), textcoords='offset points',
                                                            bbox=dict(facecolor='white', alpha=0.7,
                                                                    edgecolor='none',
                                                                    boxstyle='round,pad=0.1'),
                                                            zorder=10)
                    self.placement_artists.append(artist_text)

                if constraint.vis_end_point:
                    artist = self.ax_placement.plot(constraint.vis_end_point.x,
                                                   constraint.vis_end_point.y, 's',
                                                   color=color, markersize=8, mew=1, zorder=10)[0]
                    self.placement_artists.append(artist)
                    artist_text = self.ax_placement.annotate(f'C{idx} End',
                                                            (constraint.vis_end_point.x,
                                                             constraint.vis_end_point.y),
                                                            fontsize=7, color=color,
                                                            xytext=(5, 5), textcoords='offset points',
                                                            bbox=dict(facecolor='white', alpha=0.7,
                                                                    edgecolor='none',
                                                                    boxstyle='round,pad=0.1'),
                                                            zorder=10)
                    self.placement_artists.append(artist_text)

                # Draw pipe path (if exists) - use actual saved path
                if constraint.vis_pipe_path:
                    path = constraint.vis_pipe_path
                    path_coords_x = [(p[0] + 0.5) * self.grid_size for p in path]
                    path_coords_y = [(p[1] + 0.5) * self.grid_size for p in path]
                    line, = self.ax_placement.plot(path_coords_x, path_coords_y,
                                                  color=color, linestyle='--',
                                                  linewidth=1, alpha=0.7, zorder=9)
                    self.placement_artists.append(line)

        # Redraw placement results
        for name, data in self.placements.items():
            constraint_idx = data.get('constraint_idx', -1)
            color = self.cmap_constraints(constraint_idx / len(self.constraints)) if constraint_idx != -1 else 'black'

            if data['type'] == 'group':
                region = data['region']
                # Different style for pipe stage groups and SRAM groups
                is_pipe_stage = data.get('is_pipe_stage', False)
                is_sram = data.get('is_sram', False)
                is_sram_group = data.get('is_sram_group', False)

                if is_sram_group:  # SRAM multi-block groups - magenta color
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
                                                    fontsize=9, color='black', weight='bold',
                                                    bbox=dict(facecolor='white', alpha=0.7,
                                                            edgecolor='none',
                                                            boxstyle='round,pad=0.2'),
                                                    zorder=6)
                self.placement_artists.append(artist_text)

            elif data['type'] == 'instance':
                pos = data['position']
                artist = self.ax_placement.scatter(pos.x, pos.y, color=color, s=50, zorder=7,
                                                  edgecolors='black', linewidth=0.5)
                self.placement_artists.append(artist)
                artist_text = self.ax_placement.annotate(name, (pos.x, pos.y),
                                                        fontsize=7, color='black',
                                                        xytext=(5, 5), textcoords='offset points',
                                                        bbox=dict(facecolor='white', alpha=0.7,
                                                                edgecolor='none',
                                                                boxstyle='round,pad=0.1'),
                                                        zorder=8)
                self.placement_artists.append(artist_text)

        # Refresh the figure
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()  # Flush events to ensure figure updates


