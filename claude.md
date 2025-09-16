
# Grid-Based Pre-placement Tool with Blockage Support

## 📋 概述

這是一個用於 IC 設計的自動化預放置工具，使用網格結構和 A* 演算法來處理元件放置約束。該工具能夠讀取設計資訊、處理放置約束，並生成 Innovus TCL 腳本用於實際的晶片設計流程。

### 主要特性

- **網格化放置系統**：將晶片區域劃分為均勻網格，提供精確的放置控制
- **智能路徑規劃**：使用 A* 演算法尋找最優路徑，避開障礙物
- **多種約束類型支援**：
  - `close_to_target`：將元件放置在目標點附近
  - `pipe`：沿著路徑放置一系列元件（如 DFT scan chain）
- **視覺化除錯**：即時顯示放置過程和結果
- **完整的障礙物處理**：支援硬區塊（hard blocks）和阻擋區域（blockages）

## 🚀 快速開始

### 基本用法

```bash
python example.py <tvc.json> <io_locs.txt> <constraints.phy>
```

### 測試模式

直接運行腳本將進入測試模式，自動生成測試檔案並執行：

```bash
python example.py
```

## 📁 輸入檔案格式

### 1. TVC JSON 檔案

包含晶片設計的幾何資訊：

```json
{
  "S1_output": {
    "SoIC_A16_eTV5_root": {
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
  }
}
```

### 2. I/O 位置檔案 (io_locs.txt)

定義 I/O pad 的位置：

```
PAD_NAME {{x1 y1 x2 y2 x3 y3 x4 y4}}
PAD_IN {0 500 30 500 30 530 0 530}
PAD_OUT {2970 500 3000 500 3000 530 2970 530}
```

### 3. 約束檔案 (constraints.phy)

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

沿路徑放置元件（如 scan chain）：

```
Preplace Type: pipe
Start: start_element_name
End: end_element_name
Stage1: element1 element2 ...
Stage2: element3 element4 ...
Element Type: single
Area: 20.0
```

## 🔧 主要類別和方法

### GridBasedPlacer 類

主要的放置器類，管理整個放置流程。

#### 初始化參數

```python
placer = GridBasedPlacer(
    grid_size=50.0,           # 網格大小（微米）
    root_module_name="SoIC_A16_eTV5_root"  # 根模組名稱
)
```

#### 核心方法

| 方法 | 說明 |
|------|------|
| `load_tvc_json()` | 載入設計幾何資訊 |
| `load_io_locations()` | 載入 I/O pad 位置 |
| `load_constraints()` | 載入放置約束 |
| `build_grid_map()` | 建立網格地圖 |
| `process_constraints()` | 處理所有約束並執行放置 |
| `generate_tcl_script()` | 生成 Innovus TCL 腳本 |
| `visualize_initial_design()` | 視覺化初始設計 |

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

#### Constraint
```python
@dataclass
class Constraint:
    type: str  # 'close_to_target' 或 'pipe'
    elements: List[str]
    # ... 其他屬性依約束類型而定
```

## 🎨 視覺化功能

### 雙面板顯示

執行時會顯示兩個面板：

1. **左側：網格地圖**
   - 綠色：可用空間 (FREE)
   - 紅色：被阻擋 (BLOCKED)
   - 黃色：保留區域 (RESERVED)

2. **右側：放置結果**
   - 顯示晶片邊界、核心區域
   - 硬區塊和阻擋區域
   - I/O pads
   - 放置的元件和群組
   - 約束目標點和路徑

### 動態更新

處理每個約束時即時更新顯示，可透過 `debug_plot_interval` 參數控制更新速度。

## 🔍 網格狀態

網格可能的狀態：

| 狀態 | 值 | 說明 |
|------|-----|------|
| `GRID_FREE` | 0 | 可用於放置 |
| `GRID_BLOCKED` | 1 | 被阻擋（硬區塊、已放置元件等） |
| `GRID_RESERVED` | 2 | 臨時保留（用於群組分配） |

## 📤 輸出檔案

### 1. TCL 腳本 (dft_regs_pre_place.tcl)

生成的 Innovus 命令：

```tcl
# 群組定義
deleteInstGroup GROUP_NAME
createInstGroup GROUP_NAME -region llx lly urx ury
addInstToInstGroup GROUP_NAME { inst1 inst2 ... }

# 單一元件放置
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

### 調整視覺化速度

```python
placer.run(
    tvc_json='design.json',
    io_locs='io.txt',
    constraints='constraints.phy',
    debug_plot_interval=1.0  # 每個約束處理後暫停 1 秒
)
```

### 自訂輸出檔名

```python
placer.run(
    tvc_json='input.json',
    io_locs='io.txt',
    constraints='constraints.phy',
    output_tcl='custom_placement.tcl',
    failed_list='custom_failed.list'
)
```

## 📊 效能考量

- **網格大小**：較小的網格提供更精確的放置，但增加計算時間
- **約束數量**：大量約束可能導致處理時間增長
- **視覺化**：可以關閉視覺化以提高處理速度

## 🐛 除錯建議

1. **檢查輸入檔案格式**：確保 JSON 格式正確，座標合理
2. **觀察網格地圖**：確認障礙物正確標記
3. **檢視失敗列表**：了解哪些約束無法滿足及原因
4. **調整網格大小**：如果放置失敗過多，嘗試減小網格大小
5. **使用測試模式**：先用簡單案例測試工具功能

## 📝 注意事項

1. **座標系統**：所有座標單位為微米（μm）
2. **根模組名稱**：必須與 TVC JSON 中的鍵名匹配
3. **約束順序**：約束按檔案中的順序處理，pipe 約束可能依賴先前的放置結果
4. **Core Area**：只能在核心區域內放置元件
5. **Blockages**：會被標記為 GRID_BLOCKED，無法放置元件

## 🔄 更新歷史

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
4. 根模組名稱是否匹配

---

*此工具為 IC 設計自動化流程的一部分，旨在提高設計效率和品質。*
