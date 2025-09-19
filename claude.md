# Grid-Based Pre-placement Tool with Blockage Support
# Grid-Based Pre-placement Tool with Blockage Support

## 📋 概述

這是一個用於 IC 設計的自動化預放置工具，使用網格結構和 A* 演算法來處理元件放置約束。該工具能夠讀取設計資訊、處理放置約束，並生成 Innovus TCL 腳本用於實際的晶片設計流程。

### 主要特性

- **網格化放置系統**：將晶片區域劃分為均勻網格，提供精確的放置控制
- **智能路徑規劃**：使用 A* 演算法尋找最優路徑，避開障礙物
- **多種約束類型支援**：
  - `close_to_target`：將元件放置在目標點附近
  - `pipe`：沿著路徑放置一系列元件群組（DFT scan chain）
- **三步驟視覺化除錯**：
  - Step 1: 初始設計佈局
  - Step 2: 網格狀態地圖
  - Step 3: 即時約束處理和放置結果
- **完整的障礙物處理**：支援硬區塊（hard blocks）和阻擋區域（blockages）
- **整合式 I/O 和 MP_SENSOR 處理**：直接從 TVC JSON 載入所有設計元件

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

包含晶片設計的完整資訊，包括幾何、I/O pads、MP_SENSOR 等。注意根模組名稱必須與程式設定匹配（預設為 `uDue1/u_socss_0`）：

```json
{
  "IPS": {
    "IP_NAME": {
      "SIZE_X": 100.0,
      "SIZE_Y": 100.0
    }
  },
  "S1_output": {
    "uDue1/u_socss_0": {  // 根模組名稱，必須與程式設定匹配
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
    "uDue1/u_socss_0": {
      "GPIO": [{
        "Name": "pad_name",
        "Location": [llx, lly],
        "Orientation": "R0",
        "CellName": "PAD_CELL_TYPE"
      }]
    }
  },
  "Solution_MP_SENSOR": {
    "uDue1/u_socss_0": {
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

將元件放置在指定目標附近：

```
Preplace Type: close to target
Target Type: cell|coords|pin
Target: target_name 或 [x, y]
Element: instance_name1 instance_name2 ...
Element Type: single|instancesgroup
Area: 面積值（僅用於 instancesgroup）
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
    root_module_name="uDue1/u_socss_0"  # 根模組名稱（必須與 TVC JSON 匹配）
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

### 資料結構

#### Point
```python
@dataclass
class Point:
    x: float
    y: float
```

#### Rectangle
```python
@dataclass
class Rectangle:
    llx: float  # 左下 x
    lly: float  # 左下 y
    urx: float  # 右上 x
    ury: float  # 右上 y
```

#### Block
```python
@dataclass
class Block:
    name: str
    boundary: Rectangle
    pin_box: Optional[Rectangle] = None
```

#### Blockage
```python
@dataclass
class Blockage:
    name: str
    boundary: Rectangle
```

#### IOPad
```python
@dataclass
class IOPad:
    name: str
    boundary: Rectangle
```

#### Constraint
```python
@dataclass
class Constraint:
    type: str  # 'close_to_target' 或 'pipe'
    elements: List[str]
    stages: Optional[List[List[str]]] = None  # 僅用於 pipe，每個 stage 是一個實例列表
    # ... 其他屬性依約束類型而定
```

## 🎨 視覺化功能

### 三步驟視覺化流程

工具提供完整的三步驟視覺化流程，幫助理解和除錯：

#### Step 1: Initial Design Layout
- 顯示原始設計佈局
- 包含晶片邊界、核心區域、硬區塊、阻擋區域、I/O pads 和 MP sensors
- 顯示設計統計資訊
- 關閉視窗後繼續下一步

#### Step 2: Grid State Map
- 顯示網格地圖的初始狀態
- 使用顏色區分網格狀態：
  - 淺綠色：可用空間 (FREE)
  - 淺紅色：被阻擋 (BLOCKED)
  - 淺黃色：保留區域 (RESERVED)
- 顯示網格統計資訊
- 關閉視窗後繼續下一步

#### Step 3: Constraint Processing
- 雙面板即時顯示：
  - **左側**：網格地圖即時更新
  - **右側**：放置結果視覺化
- 處理每個約束時動態更新
- 顯示當前處理的約束（黃色高亮標籤）
- 使用不同顏色區分不同約束的放置結果
- **Pipe Stage 群組**：使用紫色半透明矩形標示，虛線邊框

### 視覺化特點

- **優化的圖例位置**：圖例放置在圖形外部或底部，避免遮擋內容
- **豐富的資訊顯示**：包含設計統計、網格使用率等資訊
- **清晰的顏色方案**：使用直觀的顏色和透明度設定
- **互動式更新**：可透過 `debug_plot_interval` 參數控制更新速度
- **區分群組類型**：一般群組與 Pipe Stage 群組使用不同視覺風格

## 🔍 網格狀態

網格可能的狀態：

| 狀態 | 值 | 說明 |
|------|-----|------|
| `GRID_FREE` | 0 | 可用於放置 |
| `GRID_BLOCKED` | 1 | 被阻擋（硬區塊、blockage、核心區域外、已放置元件） |
| `GRID_RESERVED` | 2 | 臨時保留（用於群組分配） |

## 📤 輸出檔案

### 1. TCL 腳本 (dft_regs_pre_place.tcl)

生成的 Innovus 命令：

```tcl
# Pre-placement TCL Script
# Generated by Grid-Based Placer
# Grid Size: 50.0 um
# Blockages: 3

# 群組定義（包含 close_to_target 和 pipe stage 群組）
deleteInstGroup GROUP_NAME
createInstGroup GROUP_NAME -region llx lly urx ury
addInstToInstGroup GROUP_NAME { inst1 inst2 ... }

# Pipe Stage 群組
deleteInstGroup TVC_PIPE_STAGE_0
createInstGroup TVC_PIPE_STAGE_0 -region llx lly urx ury
addInstToInstGroup TVC_PIPE_STAGE_0 { stage1_inst1 stage1_inst2 ... }

# 單一元件放置（僅來自 close_to_target 的 single 類型）
placeInstance instance_name x y -softFixed
```

### 2. 失敗列表 (failed_preplace.list)

記錄無法滿足的約束：

```
Constraint C0 (element_name): Cannot find placement near target...
Constraint C1 for elements [...]: No free space found...
```

## 🛠️ 演算法細節

### A* 路徑尋找

用於 pipe 約束，尋找避開障礙物的最短路徑：

1. 使用曼哈頓距離作為啟發函數
2. 只能在 `GRID_FREE` 的網格上移動
3. 返回最優路徑或 None（無解）

### Pipe Stage 放置

對於 pipe 約束的特殊處理：

1. 計算路徑上的放置點數量（基於 stage 數量）
2. 將路徑等分，每個 stage 群組分配一個點
3. 每個 stage 群組使用單一網格單元
4. 所有 stage 內的實例記錄在同一位置

### 最近可用網格搜尋

使用 BFS 從目標點向外搜尋最近的可用網格：

1. 從目標網格開始
2. 探索四個方向
3. 返回第一個找到的 `GRID_FREE` 網格

### 區域分配演算法

為 instancesgroup 分配矩形區域：

1. 計算所需網格數量
2. 從目標點螺旋向外搜尋
3. 尋找滿足面積要求的矩形區域
4. 標記為 `GRID_RESERVED`

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

### 調整視覺化速度

```python
placer.run(
    tvc_json='design.json',
    constraints='constraints.phy',
    debug_plot_interval=1.0  # 每個約束處理後暫停 1 秒
)
```

### 自訂輸出檔名

```python
placer.run(
    tvc_json='input.json',
    constraints='constraints.phy',
    output_tcl='custom_placement.tcl',
    failed_list='custom_failed.list'
)
```

## 📊 效能考量

- **網格大小**：較小的網格提供更精確的放置，但增加計算時間
- **約束數量**：大量約束可能導致處理時間增長
- **Pipe Stage 數量**：更多的 stage 需要更長的路徑和更多的網格
- **視覺化**：可以調整 `debug_plot_interval` 或關閉視覺化以提高處理速度

## 🐛 除錯建議

1. **檢查根模組名稱**：確保程式中的 `root_module_name` 與 TVC JSON 檔案中的鍵名完全匹配
2. **檢查輸入檔案格式**：確保 JSON 格式正確，座標合理
3. **確認 IP 尺寸定義**：確保所有 I/O pads 和 MP sensors 的 CellName 在 IPS 區塊中有對應的尺寸定義
4. **觀察三步驟視覺化**：
   - Step 1: 確認設計載入正確，包括 I/O pads 和 MP sensors
   - Step 2: 確認網格地圖正確標記障礙物
   - Step 3: 即時觀察約束處理過程，注意 Pipe Stage 群組的紫色標記
5. **檢視失敗列表**：了解哪些約束無法滿足及原因
6. **調整網格大小**：如果放置失敗過多，嘗試減小網格大小
7. **檢查 Pipe 約束**：確認起點和終點元件已經被放置或存在於設計中
8. **使用測試模式**：先用內建測試案例驗證工具功能

## 📝 注意事項

1. **根模組名稱**：預設為 `"uDue1/u_socss_0"`，必須與 TVC JSON 中的鍵名匹配
2. **座標系統**：所有座標單位為微米（μm）
3. **約束順序**：先處理所有 close_to_target，再處理 pipe 約束
4. **Pipe Stage 群組**：每個 stage 使用單一網格單元，適合小型群組
5. **Core Area**：只能在核心區域內放置元件
6. **Blockages**：會被標記為 GRID_BLOCKED，無法放置元件
7. **I/O Pads 和 MP Sensors**：從 TVC JSON 自動載入，作為可參考的目標但不可覆蓋
8. **視覺化視窗**：每個步驟結束後需手動關閉視窗才能繼續

## 🔄 更新歷史

### 最新更新 (Pipe Stage Groups)
- **Pipe 約束改進**：移除 `Element Type: single`，預設為 `instancesgroup`
- **Stage 群組化**：每個 pipe stage 現在作為獨立群組處理
- **簡化語法**：pipe 約束不再需要指定 Element Type 或 Area
- **視覺化增強**：Pipe Stage 群組使用紫色和虛線邊框區分
- **TCL 輸出優化**：所有群組（包括 pipe stages）都生成 group 命令

### 先前更新
- **簡化輸入檔案**：移除獨立的 I/O 位置檔案，整合到 TVC JSON
- **新增 MP_SENSOR 支援**：自動從 Solution_MP_SENSOR 載入並處理
- **改進 I/O pads 處理**：從 Solution_GPIO 自動載入
- **更新預設根模組名稱**：改為 `uDue1/u_socss_0`
- 改進三步驟視覺化流程，提供更清晰的除錯體驗
- 優化圖例位置和資訊顯示
- 支援 blockages 處理
- 新增 pipe 約束類型
- 改進視覺化系統
- 優化 A* 演算法效能
- 支援 instancesgroup 類型

## 📧 技術支援

如遇到問題，請檢查：

1. Python 版本 >= 3.8
2. 必要套件：numpy, matplotlib
3. 輸入檔案格式是否正確
4. 根模組名稱是否匹配（預設：`uDue1/u_socss_0`）
5. IPS 區塊中是否包含所有必要的尺寸定義
6. Solution_GPIO 和 Solution_MP_SENSOR 區塊是否存在且格式正確
7. Pipe 約束的起點和終點是否有效
8. 視覺化視窗是否正確關閉以繼續流程

---

*此工具為 IC 設計自動化流程的一部分，旨在提高設計效率和品質。*
