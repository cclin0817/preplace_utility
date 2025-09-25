#!/usr/bin/env python3
"""
Grid-Based Pre-placement Tool with Blockage Support
Entry point for the application
"""

import sys
from grid_placer import GridBasedPlacer
from testcase import create_test_files_with_blockages


def main():
    """Main entry point for the application"""
    # Default root module name
    ROOT_MODULE_NAME = "SoIC_A16_eTV5_root"

    if len(sys.argv) == 3:
        # Normal mode with command line arguments
        placer = GridBasedPlacer(grid_size=10.0, root_module_name=ROOT_MODULE_NAME)
        placer.run(
            tvc_json=sys.argv[1],
            constraints=sys.argv[2]
        )
    else:
        # Test mode
        print("Usage: python main.py <tvc.json> <constraints.phy>")
        print(f"\nRunning test mode with blockages and default root module: '{ROOT_MODULE_NAME}'...")

        # Create test files
        create_test_files_with_blockages()

        # Execute test
        placer = GridBasedPlacer(grid_size=50.0, root_module_name=ROOT_MODULE_NAME)
        placer.run(
            tvc_json='test_tvc_blockage.json',
            constraints='test_constraints.phy'
        )


if __name__ == "__main__":
    main()
