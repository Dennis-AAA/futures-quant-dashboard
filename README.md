# 五品种期货与期权曲面量化指标回测看板 (CL / GC / HG / ES / NQ · 2014–2026)

本项目包含基于 **Databento (`GLBX.MDP3`) 连续主力期货日频数据** 与 **Black-76 期权隐含波动率曲面数据**（2014.01–2026.10）构建的「A+B混合六步法」全品种量化指标筛选与逐笔交割互动复盘看板。

## 在线预览文件说明

- **`index.html`**（主看板）：五品种（WTI原油 `CL`、COMEX黄金 `GC`、COMEX铜 `HG`、标普500 `ES`、纳指100 `NQ`）交互式历史信号亮灯复盘看板（零外部CDN依赖，支持离线/国内网络秒开）。
- **`wti_signals_comparison_dashboard.html`**：WTI原油报告A（Walk-Forward）与报告B（全样本分位数）历史亮灯位置双图同步对比复盘。
- **`four_futures_backtest_results_2014_2026.csv`**：全品种 Q1/Q2/Q3 指标筛选成绩单与样本外实测对比表。
- **`multi_asset_indicator_screener.py`**：多品种「A+B混合六步法」Python 源码引擎。

## 如何一键开启 GitHub Pages 在线访问

1. 在 GitHub 新建一个仓库（例如 `futures-quant-dashboard`，设为 `Public`）。
2. 点击 **Add file → Upload files**，将本压缩包解压后的 `index.html`、`wti_signals_comparison_dashboard.html` 等文件拖拽上传并点击 **Commit changes**。
3. 进入仓库顶部的 **Settings → Pages**（左侧菜单），在 **Build and deployment → Branch** 下选择 `main`（或 `master`）分支与 `/ (root)` 目录，点击 **Save**。
4. 等待约 30–60 秒后，直接访问 `https://<你的GitHub用户名>.github.io/futures-quant-dashboard/` 即可在手机或电脑浏览器中流畅打开全景交互看板。
