
def create_test_files_with_blockages():
    """建立包含 blockages 的測試檔案 - Updated for pipe stage groups"""
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
                    },
                    {
                        "Name": "PAD_OUT",
                        "Location": [2970, 500],
                        "Orientation": "R0",
                        "CellName": "PAD_CELL",
                        "Legend_key": "PAD_CELL"
                    },
                    {
                        "Name": "PAD_TOP",
                        "Location": [1000, 1970],
                        "Orientation": "R0",
                        "CellName": "PAD_CELL",
                        "Legend_key": "PAD_CELL"
                    },
                    {
                        "Name": "PAD_BOT",
                        "Location": [1000, 0],
                        "Orientation": "R0",
                        "CellName": "PAD_CELL",
                        "Legend_key": "PAD_CELL"
                    }
                ]
            }
        },
        # 新增 Solution_MP_SENSOR 區塊
        "Solution_MP_SENSOR": {
            "SoIC_A16_eTV5_root": {
                "MP_SENSOR": [
                    {
                        "Name": "mp_sensor_1",
                        "Location": [1500, 500],
                        "CellName": "MP_SENSOR_TYPE1"
                    },
                    {
                        "Name": "mp_sensor_2",
                        "Location": [1800, 1500],
                        "CellName": "MP_SENSOR_TYPE1"
                    }
                ]
            }
        },
        # 新增 IPS 區塊以提供 I/O Pad 和 MP Sensor 的尺寸資訊
        "IPS": {
            "PVDD1CODCDGM_H": {"SIZE_X": 30.0, "SIZE_Y": 52.91},
            "PVDD08CODCDGM_H": {"SIZE_X": 30.0, "SIZE_Y": 52.91},
            "PVDD1204CODCDGM_H": {"SIZE_X": 30.0, "SIZE_Y": 52.91},
            "VDD_CORNER": {"SIZE_X": 30.0, "SIZE_Y": 30.0}, # 假設角落 I/O Pad 尺寸
            "PAD_CELL": {"SIZE_X": 30.0, "SIZE_Y": 30.0},
            "MP_SENSOR_TYPE1": {"SIZE_X": 50.0, "SIZE_Y": 50.0},  # MP sensor 尺寸
            # 也可以包含 Blocks 的 IP 尺寸
            "BLOCK1": {"SIZE_X": 500.0, "SIZE_Y": 500.0},
            "BLOCK2": {"SIZE_X": 500.0, "SIZE_Y": 500.0}
        }
    }

    with open('test_tvc_blockage.json', 'w') as f:
        json.dump(tvc_data, f, indent=2)

    # 約束檔案 - 測試 close_to_target 然後 pipe 的順序
    # MODIFIED: Remove "Element Type: single" from pipe constraints
    with open('test_constraints.phy', 'w') as f:
        # First: close_to_target constraints with module type
        f.write("Preplace Type: close to target\n")
        f.write("Target Type: pin\n")
        f.write("Target: u_block1\n")
        f.write("Element: u_test/inst1\n")
        f.write("Element Type: single\n")
        f.write("Area: 20.0\n\n")

        f.write("Preplace Type: close to target\n")
        f.write("Target Type: cell\n")
        f.write("Target: PAD_OUT\n")
        f.write("Element: u_test/inst2\n")
        f.write("Element Type: single\n")
        f.write("Area: 20.0\n\n")

        f.write("Preplace Type: close to target\n")
        f.write("Target Type: coords\n")
        f.write("Target: [150.0, 150.0]\n")
        f.write("Element: u_test/inst3\n")
        f.write("Element Type: single\n")
        f.write("Area: 20.0\n\n")

        f.write("Preplace Type: close to target\n")
        f.write("Target Type: coords\n")
        f.write("Target: [1500.0, 1000.0]\n")
        f.write("Element: u_test/inst4\n")
        f.write("Element Type: single\n")
        f.write("Area: 20.0\n\n")

        f.write("Preplace Type: close to target\n")
        f.write("Target Type: cell\n")
        f.write("Target: u_block2\n")
        f.write("Element: u_test/inst5\n")
        f.write("Element Type: single\n")
        f.write("Area: 20.0\n\n")

        # Test MP sensor as target
        f.write("Preplace Type: close to target\n")
        f.write("Target Type: cell\n")
        f.write("Target: mp_sensor_1\n")
        f.write("Element: u_test/inst6\n")
        f.write("Element Type: single\n")
        f.write("Area: 20.0\n\n")

        # Test module type (treated as instancesgroup)
        f.write("Preplace Type: close to target\n")
        f.write("Target Type: coords\n")
        f.write("Target: [1300.0, 1300.0]\n")
        f.write("Element: GROUP_A/reg_0 GROUP_A/reg_1 GROUP_A/reg_2\n")
        f.write("Element Type: module\n")  # module type should behave like instancesgroup
        f.write("Area: 50000.0\n\n")

        # Then: pipe constraints - NO "Element Type: single" line
        f.write("Preplace Type: pipe\n")
        f.write("Start: PAD_BOT\n")  # Start from I/O pad
        f.write("End: PAD_TOP\n")    # End at another I/O pad
        f.write("Stage1: u_dft/s_0 u_dft/s_1 u_dft/s_2\n")
        f.write("Stage2: u_dft/s_3 u_dft/s_4 u_dft/s_5\n\n")  # No Element Type line

        f.write("Preplace Type: pipe\n")
        f.write("Start: u_test/inst1\n")  # Start from previously placed instance
        f.write("End: u_test/inst5\n")    # End at previously placed instance
        f.write("Stage1: u_dft_sub/s_0 u_dft_sub/s_1 u_dft_sub/s_2\n\n")  # No Element Type line

        f.write("Preplace Type: pipe\n")
        f.write("Start: GROUP_A/reg_0\n")  # Start from instance in group
        f.write("End: u_test/inst2\n")     # End at single instance
        f.write("Stage1: u_pipe/p_0 u_pipe/p_1\n")
        f.write("Stage2: u_pipe/p_2 u_pipe/p_3\n\n")  # No Element Type line

        # Test failure case
        f.write("Preplace Type: close to target\n")
        f.write("Target Type: coords\n")
        f.write("Target: [100.0, 100.0]\n")  # Outside Core Area
        f.write("Element: u_test/inst_fail\n")
        f.write("Element Type: single\n")
        f.write("Area: 20.0\n\n")

    print("Test files created with pipe stage groups support!")
    print("Pipe constraints now create groups for each stage (one grid per stage).")

