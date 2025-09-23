# Grid-Based Pre-placement Tool with Blockage Support
### 設計理念

這種改進的方法更符合實際 IC 設計需求：

1. **保持連接完整性**：DFT scan chain 必須連接已經確定位置的元件（如 I/O pads、功能模塊），不應改變這些元件的位置
2. **現實的繞線模型**：實際佈線可以穿過放置群組的上方（不同金屬層），因此路徑規劃允許穿過 RESERVED 區域
3. **分離關注點**：路徑規劃（routing）與元件放置（placement）是兩個不同的步驟，不應混淆# Grid-Based Pre-placement Tool with Blockage Support

## 📋 概述

這是一個用於 IC 設計的自動化預放置工具，使用網格結構和 A* 演算法來處理元件放置約束。該工具能夠讀取設計資訊、處理放置約束，並生成 Innovus TCL 腳本用於實際的晶片設計流程。

### 主要特性

- **網格化放置系統**：將晶片區域劃分為均勻網格，提供精確的放置控制
- **智能路徑規劃**：使用 A* 演算法尋找最優路徑，避開障礙物
- **多種約束類型支援**：
  - `close_to_target`：將元件放置在目標點附近
    - 支援 SRAM 特殊放置（根據 pin edge 自動決定放置位置）
    - 支援 SRAM_GROUP 放置（在多個區塊的邊界框內尋找最大可用矩形）
  - `pipe`：沿著路徑放置一系列元件群組（DFT scan chain）
- **三步驟視覺化除錯**：
  - Step 1: 初始設計佈局
  - Step 2: 網格狀態地圖
  - Step 3: 即時約束處理和放置結果
- **完整的障礙物處理**：支援硬區塊（hard blocks）和阻擋區域（blockages）
- **整合式 I/O 和 MP_SENSOR 處理**：直接從 TVC JSON 載入所有設計元件
- **SRAM 智能放置**：
  - 單一目標區塊：根據 pin box 位置自動決定最佳放置邊
  - 多個目標區塊：在邊界框內找出最大可用矩形區域

## 🚀 快速開始

### 基本用法

```bash
python example.py <tvc.json> <constraints.phy>
```

### 測試模式

直接運行腳本將進入測試模式，自動生成測試檔案並執行：

```bash
python example.py
```

## 📁 輸入檔案格式

### 1. TVC JSON 檔案

包含晶片設計的完整資訊，包括幾何、I/O pads、MP_SENSOR 等。注意根模組名稱必須與程式設定匹配（預設為 `SoIC_A16_eTV5_root`）：

```json
{
  "IPS": {
    "IP_NAME": {
      "SIZE_X": 100.0,
      "SIZE_Y": 100.0
    }
  },
  "S1_output": {
    "SoIC_A16_eTV5_root": {  // 根模組名稱，必須與程式設定匹配
      "TOPShape": {
        "Name": "chip_name",
        "DieCoords": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
        "CoreCoords": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
        "Blockages": [
          [[x1,y1], [x2,y2], ...],  // 阻擋區域座標
          ...
        ]
      },
      "BlockShapes": [{
        "BlockName": "block_name",
        "DesignName": "design_name",
        "Coords": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
        "PinCoords": [[[x1,y1], [x2,y2]], ...],
        "Attribute": "Module"
      }]
    }
  },
  "Solution_GPIO": {
    "SoIC_A16_eTV5_root": {
      "GPIO": [{
        "Name": "pad_name",
        "Location": [llx, lly],
        "Orientation": "R0",
        "CellName": "PAD_CELL_TYPE"
      }]
    }
  },
  "Solution_MP_SENSOR": {
    "SoIC_A16_eTV5_root": {
      "MP_SENSOR": [{
        "Name": "sensor_name",
        "Location": [llx, lly],
        "CellName": "SENSOR_TYPE"
      }]
    }
  }
}
```

#### 重要組成部分：

- **IPS**：定義所有 IP 元件的尺寸，用於計算 I/O pads 和 MP_SENSOR 的實際大小
- **S1_output**：包含晶片的主要幾何資訊
- **Solution_GPIO**：定義所有 I/O pads 的位置和類型
- **Solution_MP_SENSOR**：定義所有 MP sensor 的位置和類型

### 2. 約束檔案 (constraints.phy)

#### Close-to-Target 約束

將元件群組放置在指定目標附近（注意：不再支援 single element type）：

```
Preplace Type: close to target
Target Type: cell|coords|pin|sram|sram_group
Target: target_name 或 [x, y] 或 space-separated block names (for sram_group)
Element: instance_name1 instance_name2 ...
Element Type: instancesgroup|module
Area: 面積值
```

##### 支援的 Target Types

1. **cell**: 放置在指定元件（block/IO pad/MP sensor）附近
   ```
   Target Type: cell
   Target: u_cpu_core
   ```

2. **coords**: 放置在指定座標附近
   ```
   Target Type: coords
   Target: [1500.0, 1000.0]
   ```

3. **pin**: 放置在指定 pin 附近
   ```
   Target Type: pin
   Target: u_block1/pin_name
   ```

4. **sram**: 智能 SRAM 放置（單一目標區塊）
   - 自動分析目標區塊的 pin box 位置
   - 決定最佳放置邊（top/bottom/left/right）
   - 在對應邊創建 SRAM 群組
   ```
   Target Type: sram
   Target: u_cpu_core
   Element: sram_inst_0 sram_inst_1 sram_inst_2
   Element Type: instancesgroup
   Area: 50000.0
   ```

5. **sram_group**: 多區塊 SRAM 放置（新功能）
   - 計算多個目標區塊的邊界框
   - 在邊界框內尋找最大可用矩形區域
   - 適合需要連接多個區塊的記憶體群組
   ```
   Target Type: sram_group
   Target: u_block1 u_cpu_core u_dsp_block
   Element: multi_sram_0 multi_sram_1 multi_sram_2
   Element Type: instancesgroup
   Area: 80000.0
   ```

#### Pipe 約束

沿路徑放置元件群組（如 scan chain），每個 Stage 會被當作一個群組：

```
Preplace Type: pipe
Start: start_element_name
End: end_element_name
Stage1: element1 element2 ...
Stage2: element3 element4 ...
```

注意：
- pipe 類型的預設 `Element Type` 為 `instancesgroup`
- 不需要指定 `Element Type` 或 `Area`
- 每個 Stage 會被放置為一個群組，使用單一網格單元
- 處理順序：先處理所有 close_to_target 約束，再處理 pipe 約束

## 🔧 主要類別和方法

### GridBasedPlacer 類

主要的放置器類，管理整個放置流程。

#### 初始化參數

```python
placer = GridBasedPlacer(
    grid_size=50.0,           # 網格大小（微米）
    root_module_name="SoIC_A16_eTV5_root"  # 根模組名稱（必須與 TVC JSON 匹配）
)
```

#### 核心方法

| 方法 | 說明 |
|------|------|
| `load_tvc_json()` | 載入設計幾何資訊、I/O pads、MP sensors |
| `load_constraints()` | 載入放置約束 |
| `build_grid_map()` | 建立網格地圖 |
| `process_constraints()` | 處理所有約束並執行放置 |
| `generate_tcl_script()` | 生成 Innovus TCL 腳本 |
| `visualize_initial_design()` | 視覺化初始設計（步驟 1） |
| `visualize_grid_state()` | 視覺化網格狀態（步驟 2） |
| `determine_pin_edge()` | 判斷區塊的 pin box 位於哪一邊 |
| `allocate_sram_region()` | 根據 pin edge 分配 SRAM 區域 |
| `find_max_rectangle_in_region()` | 在邊界框內尋找最大可用矩形（新增） |

### 資料結構

#### Constraint
```python
@dataclass
class Constraint:
    type: str  # 'close_to_target' 或 'pipe'
    elements: List[str]
    target_type: Optional[str] = None  # 'cell', 'coords', 'pin', 'sram', 'sram_group'
    target_blocks: Optional[List[str]] = None  # 用於 sram_group，目標區塊列表
    stages: Optional[List[List[str]]] = None  # 僅用於 pipe
    element_type: str = 'instancesgroup'  # 'instancesgroup' 或 'module'（不再支援 'single'）
    area: float = 20.0
    # ... 其他視覺化相關屬性
```

## 🎨 視覺化功能

### 三步驟視覺化流程

工具提供完整的三步驟視覺化流程，幫助理解和除錯：

#### Step 1: Initial Design Layout
- 顯示原始設計佈局
- 包含晶片邊界、核心區域、硬區塊、阻擋區域、I/O pads 和 MP sensors
- 顯示設計統計資訊

#### Step 2: Grid State Map
- 顯示網格地圖的初始狀態
- 使用顏色區分網格狀態：
  - 淺綠色：可用空間 (FREE)
  - 淺紅色：被阻擋 (BLOCKED)
  - 淺黃色：保留區域 (RESERVED)

#### Step 3: Constraint Processing
- 雙面板即時顯示：
  - **左側**：網格地圖即時更新
  - **右側**：放置結果視覺化
- 使用不同顏色區分不同類型的群組：
  - **一般群組**：使用約束對應的顏色，半透明
  - **Pipe Stage 群組**：紫色半透明矩形，虛線邊框
  - **SRAM 群組**（單一區塊）：青色（cyan）半透明矩形，實線邊框
  - **SRAM Multi-Block 群組**：洋紅色（magenta）半透明矩形，實線邊框

## 📤 輸出檔案

### 1. TCL 腳本 (dft_regs_pre_place.tcl)

生成的 Innovus 命令（所有元件都作為群組放置）：

```tcl
# Pre-placement TCL Script
# Generated by Grid-Based Placer
# Grid Size: 10.0 um

# 一般群組
deleteInstGroup TVC_INST_GROUP_0
createInstGroup TVC_INST_GROUP_0 -region llx lly urx ury
addInstToInstGroup TVC_INST_GROUP_0 { inst1 inst2 ... }

# Pipe Stage 群組
deleteInstGroup TVC_PIPE_STAGE_0
createInstGroup TVC_PIPE_STAGE_0 -region llx lly urx ury
addInstToInstGroup TVC_PIPE_STAGE_0 { stage1_inst1 stage1_inst2 ... }

# SRAM 群組（單一區塊目標）
deleteInstGroup TVC_SRAM_GROUP_0
createInstGroup TVC_SRAM_GROUP_0 -region llx lly urx ury
addInstToInstGroup TVC_SRAM_GROUP_0 { sram_inst_0 sram_inst_1 ... }

# SRAM Multi-Block 群組（多區塊目標）
deleteInstGroup TVC_SRAM_MULTI_GROUP_0
createInstGroup TVC_SRAM_MULTI_GROUP_0 -region llx lly urx ury
addInstToInstGroup TVC_SRAM_MULTI_GROUP_0 { multi_sram_0 multi_sram_1 ... }
```

### 2. 失敗列表 (failed_preplace.list)

記錄無法滿足的約束：

```
Constraint C0 (elements: [...]): Cannot find placement near target...
Constraint C1 for elements [...]: No free space found...
```

## 🛠️ 演算法細節

### A* 路徑尋找

用於 pipe 約束，尋找避開硬障礙物的最優路徑：

1. 使用曼哈頓距離作為啟發函數
2. 可以通過 `GRID_FREE` 和 `GRID_RESERVED` 網格（允許穿過已放置的群組）
3. 只避開 `GRID_BLOCKED`（硬區塊、blockage、核心區域外）
4. 返回最優路徑或 None（無解）

### SRAM 放置策略

#### 單一區塊目標（Target Type: sram）

1. **Pin Edge 判定**：計算 pin box 到區塊四邊的距離，選擇最近的邊
2. **區域計算**：根據 pin edge 決定 SRAM 群組的形狀和位置
3. **自動優化**：確保最短連線長度，減少繞線複雜度

#### 多區塊目標（Target Type: sram_group）

1. **邊界框計算**：計算所有目標區塊的邊界矩形
2. **最大矩形搜尋**：在邊界框內使用直方圖演算法尋找最大可用矩形
3. **位置優化**：選擇面積最大且最接近邊界框中心的矩形
4. **彈性配置**：自動適應不同的區塊配置和可用空間

### Pipe Stage 放置策略

1. **路徑規劃**：使用 A* 找出從起點到終點的路徑
2. **Stage 分配**：將路徑等分，每個分段點作為 stage 群組位置
3. **放置驗證**：檢查並調整到最近的可用網格

## ⚙️ 進階設定

### 自訂網格大小

```python
placer = GridBasedPlacer(grid_size=100.0)  # 使用 100 微米網格
```

### 自訂根模組名稱

```python
placer = GridBasedPlacer(
    grid_size=50.0,
    root_module_name="your_custom_module_name"  # 必須與 TVC JSON 匹配
)
```

## 📊 效能考量

- **網格大小**：較小的網格提供更精確的放置，但增加計算時間
- **約束數量**：大量約束可能導致處理時間增長
- **SRAM 群組**：
  - 單一區塊：Pin edge 判定為 O(1) 操作
  - 多區塊：最大矩形搜尋複雜度較高，但通常仍很快速
- **路徑規劃**：改進的 A* 演算法允許穿過 RESERVED 區域，提高效率

## 🐛 除錯建議

1. **檢查根模組名稱**：確保程式中的 `root_module_name` 與 TVC JSON 檔案中的鍵名完全匹配
2. **檢查輸入檔案格式**：確保 JSON 格式正確，座標合理
3. **確認 Element Type**：只使用 `instancesgroup` 或 `module`，不再支援 `single`
4. **觀察三步驟視覺化**：
   - Step 1: 確認設計載入正確
   - Step 2: 確認網格地圖正確標記障礙物
   - Step 3: 觀察約束處理過程，注意不同顏色的群組類型
5. **檢查 SRAM 約束**：
   - 單一區塊：確認目標區塊存在且有 pin box
   - 多區塊：確認所有目標區塊都存在
6. **使用測試模式**：先用內建測試案例驗證工具功能

## 📝 注意事項

1. **根模組名稱**：預設為 `"SoIC_A16_eTV5_root"`，必須與 TVC JSON 中的鍵名匹配
2. **座標系統**：所有座標單位為微米（μm）
3. **約束順序**：先處理所有 close_to_target，再處理 pipe 約束
4. **Element Type**：
   - **不再支援 `single` 類型**
   - 所有元件都作為群組（`instancesgroup` 或 `module`）放置
   - Pipe 約束自動使用 `instancesgroup`
5. **SRAM 放置**：
   - 單一區塊：生成 `TVC_SRAM_GROUP_x` 群組
   - 多區塊：生成 `TVC_SRAM_MULTI_GROUP_x` 群組
6. **群組命名**：
   - 一般群組：`TVC_INST_GROUP_x`
   - Pipe Stage：`TVC_PIPE_STAGE_x`
   - SRAM（單一）：`TVC_SRAM_GROUP_x`
   - SRAM（多區塊）：`TVC_SRAM_MULTI_GROUP_x`

## 🔄 更新歷史

### 最新更新 (Remove Single Element Type & Add SRAM_GROUP)
- **移除 single element type**：所有元件現在都作為群組放置
- **新增 sram_group target type**：支援多區塊邊界框內的 SRAM 放置
- **新增最大矩形搜尋演算法**：`find_max_rectangle_in_region()` 方法
- **視覺化增強**：新增洋紅色（magenta）標示多區塊 SRAM 群組
- **簡化約束語法**：移除不必要的 element type 選項
- **TCL 輸出優化**：所有輸出都是群組命令，沒有個別實例放置

### 先前更新
- 新增 SRAM target type（單一區塊）
- 改進 Pipe 路徑尋找演算法
- 新增 MP_SENSOR 支援
- 支援 blockages 處理
- 三步驟視覺化流程

---

*此工具為 IC 設計自動化流程的一部分，旨在提高設計效率和品質。*
