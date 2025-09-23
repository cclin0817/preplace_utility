
def create_test_files_with_blockages():
    """建立包含 blockages 的測試檔案 - Updated to remove 'single' element type"""
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
                    "PinCoords": [[[2450, 1450], [2490, 1490]]], # 右側 pin box
                    "Attribute": "Module"
                }, {
                    "BlockName": "u_cpu_core",
                    "DesignName": "CPU_CORE",
                    "Coords": [[800, 1200], [1300, 1200], [1300, 1600], [800, 1600]],
                    "PinCoords": [[[850, 1550], [950, 1590]]], # 頂部 pin box for SRAM test
                    "Attribute": "Module"
                }, {
                    "BlockName": "u_dsp_block",
                    "DesignName": "DSP_BLOCK",
                    "Coords": [[1600, 400], [1900, 400], [1900, 700], [1600, 700]],
                    "PinCoords": [[[1550, 550], [1590, 650]]], # 左側 pin box
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
        # Solution_GPIO 區塊
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
        # Solution_MP_SENSOR 區塊
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
        # IPS 區塊以提供 I/O Pad 和 MP Sensor 的尺寸資訊
        "IPS": {
            "PVDD1CODCDGM_H": {"SIZE_X": 30.0, "SIZE_Y": 52.91},
            "PVDD08CODCDGM_H": {"SIZE_X": 30.0, "SIZE_Y": 52.91},
            "PVDD1204CODCDGM_H": {"SIZE_X": 30.0, "SIZE_Y": 52.91},
            "VDD_CORNER": {"SIZE_X": 30.0, "SIZE_Y": 30.0}, # 假設角落 I/O Pad 尺寸
            "PAD_CELL": {"SIZE_X": 30.0, "SIZE_Y": 30.0},
            "MP_SENSOR_TYPE1": {"SIZE_X": 50.0, "SIZE_Y": 50.0},  # MP sensor 尺寸
            # Blocks 的 IP 尺寸
            "BLOCK1": {"SIZE_X": 500.0, "SIZE_Y": 500.0},
            "BLOCK2": {"SIZE_X": 500.0, "SIZE_Y": 500.0},
            "CPU_CORE": {"SIZE_X": 500.0, "SIZE_Y": 400.0},
            "DSP_BLOCK": {"SIZE_X": 300.0, "SIZE_Y": 300.0}
        }
    }

    with open('test_tvc_blockage.json', 'w') as f:
        json.dump(tvc_data, f, indent=2)

    with open('test_constraints.phy', 'w') as f:
        f.write("Preplace Type: close to target\n")
        f.write("Target Type: pin\n")
        f.write("Target: u_block1\n")
        f.write("Element: u_test/inst1\n")
        f.write("Element Type: instancesgroup\n")
        f.write("Area: 100.0\n\n")  # Need area for groups

        f.write("Preplace Type: close to target\n")
        f.write("Target Type: cell\n")
        f.write("Target: PAD_OUT\n")
        f.write("Element: u_test/inst2\n")
        f.write("Element Type: instancesgroup\n")
        f.write("Area: 100.0\n\n")

        f.write("Target Type: coords\n")
        f.write("Target: [1500.0, 1000.0]\n")
        f.write("Element: u_test/reg_0 u_test/reg_1 u_test/reg_2 u_test/reg_3\n")
        f.write("Element Type: instancesgroup\n")
        f.write("Area: 10000.0\n\n")

        # Test SRAM placement (single target block)
        f.write("Preplace Type: close to target\n")
        f.write("Target Type: sram\n")
        f.write("Target: u_cpu_core\n")
        f.write("Element: sram_inst_0 sram_inst_1 sram_inst_2 sram_inst_3\n")
        f.write("Element Type: instancesgroup\n")
        f.write("Area: 50000.0\n\n")

        # Test SRAM_GROUP placement (multiple target blocks)
        f.write("Preplace Type: close to target\n")
        f.write("Target Type: sram_group\n")
        f.write("Target: u_block1 u_cpu_core u_dsp_block\n")  # Multiple blocks
        f.write("Element: multi_sram_0 multi_sram_1 multi_sram_2 multi_sram_3 multi_sram_4\n")
        f.write("Element Type: instancesgroup\n")
        f.write("Area: 80000.0\n\n")

        # Test module type (behaves like instancesgroup)
        f.write("Preplace Type: close to target\n")
        f.write("Target Type: coords\n")
        f.write("Target: [1300.0, 1300.0]\n")
        f.write("Element: GROUP_A/reg_0 GROUP_A/reg_1 GROUP_A/reg_2\n")
        f.write("Element Type: module\n")  # module type behaves like instancesgroup
        f.write("Area: 50000.0\n\n")

        # Test close to MP sensor
        f.write("Preplace Type: close to target\n")
        f.write("Target Type: cell\n")
        f.write("Target: mp_sensor_1\n")
        f.write("Element: u_test/sensor_reg_0 u_test/sensor_reg_1\n")
        f.write("Element Type: instancesgroup\n")
        f.write("Area: 5000.0\n\n")

        # Pipe constraints - no Element Type needed
        f.write("Preplace Type: pipe\n")
        f.write("Start: PAD_BOT\n")
        f.write("End: PAD_TOP\n")
        f.write("Stage1: u_dft/s_0 u_dft/s_1 u_dft/s_2\n")
        f.write("Stage2: u_dft/s_3 u_dft/s_4 u_dft/s_5\n")
        f.write("Stage3: u_dft/s_6 u_dft/s_7\n\n")

        f.write("Preplace Type: pipe\n")
        f.write("Start: u_test/inst1\n")  # Start from previously placed instance
        f.write("End: u_test/inst2\n")     # End at previously placed instance
        f.write("Stage1: u_dft_sub/s_0 u_dft_sub/s_1 u_dft_sub/s_2\n\n")

        f.write("Preplace Type: pipe\n")
        f.write("Start: GROUP_A/reg_0\n")  # Start from instance in group
        f.write("End: sram_inst_0\n")      # End at SRAM instance
        f.write("Stage1: u_pipe/p_0 u_pipe/p_1\n")
        f.write("Stage2: u_pipe/p_2 u_pipe/p_3\n")
        f.write("Stage3: u_pipe/p_4 u_pipe/p_5 u_pipe/p_6\n\n")

        # Test another SRAM placement
        f.write("Preplace Type: close to target\n")
        f.write("Target Type: sram\n")
        f.write("Target: u_dsp_block\n")
        f.write("Element: dsp_sram_0 dsp_sram_1 dsp_sram_2\n")
        f.write("Element Type: instancesgroup\n")
        f.write("Area: 30000.0\n\n")

        # Test failure case - area too large
        f.write("Preplace Type: close to target\n")
        f.write("Target Type: coords\n")
        f.write("Target: [2700.0, 1700.0]\n")  # Near edge of core area
        f.write("Element: u_test/large_group_0 u_test/large_group_1\n")
        f.write("Element Type: instancesgroup\n")
        f.write("Area: 1000000.0\n\n")  # Too large to fit

    print("Test files created with updated constraints!")
    print("Changes:")
    print("  - Removed all 'single' element type (now use 'instancesgroup')")
    print("  - Added SRAM target type test (single block)")
    print("  - Added SRAM_GROUP target type test (multiple blocks)")
    print("  - Pipe constraints create stage groups")
    print("  - All placements now create groups, no individual instance placement")
