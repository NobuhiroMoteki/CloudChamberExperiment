# Cloud Chamber Experiment（日本語の案内）

English: [README.md](README.md)

減圧式の雲チェンバーで、減圧中のレーザー透過率 Tr(t) から、試料空気の**水蒸気混合比 qw** と**雲凝結核（エアロゾル）数濃度 N** を 1 分ごとに推定する装置のソフトウェアと文書です（環境実習）。

## 論文とデータ

- 論文：N. Moteki, *Working-fluid-free measurement of particle number concentration and water vapor from the laser-extinction curve of an expansion cloud by simulation-based inference*（Journal of Aerosol Science に投稿）。原稿と補足資料は `Paper/JAerosolSci/`、図表は `Python/paper_*.py` で作ります。
- データ（2026-10-31〜11-02 の評価期間）：Zenodo、https://doi.org/10.5281/zenodo.24070205
- 論文の結果を再現する手順は、英語の [README.md](README.md) にあります。

## 現状（2026-11-09）

- 計測制御は LabVIEW から Python に移行済みです。構成は NI USB-6210、PyQtGraph の GUI、60 秒サイクルです。
- 推定器は **Tr 形状の SBI（版4、`sbi_shape4.py`）** です。光検出器に入る前方散乱光を補正した消散効率を使い、圧力の曲線は記録した曲線から取ります。気圧センサーの補正は使いません。
- 参照測定器との比較（2026-10-31〜11-02、52 時間、3114 サイクル、学習に使っていない日の推定）：

  | 量 | 参照 | 推定 ÷ 参照（中央値、四分位範囲） | 相関係数 |
  |---|---|---|---|
  | qw | 鏡面式露点計 DPH-203B | 0.997（0.987〜1.007） | 0.992 |
  | N | CPC TSI 3022A | 0.995（0.950〜1.046） | 0.997 |

## フォルダ構成

| フォルダ | 内容 |
|---|---|
| `Python/cloudchamber/` | 計測制御プログラムの本体。DAQ、バルブ制御、GUI、順モデル（`activation.py`）、SBI（`sbi_shape3.py` など）、CPC の読み取り（`cpc.py`） |
| `Python/models/` | 学習済みの事後分布（GUI の既定は `npe_shape4_30000.pt`）と、論文の感度試験の結果 |
| `Python/MANUAL.md` | **操作マニュアル**（配線、起動、点検、出力ファイルの列、CPC の記録、困ったとき） |
| `Python/run_gui.bat` | GUI の起動用 |
| `Python/run_sbi_shape4.py`、`Python/reanalyze.py` | SBI（版4）の学習と、記録したサイクルの推定のやり直し |
| `Python/paper_*.py` | 論文の図・表・数値を作るスクリプト |
| `Paper/JAerosolSci/` | 論文の原稿と補足資料（LuaLaTeX） |
| `Python/cpc_logger.py`、`check_daq.py` | GUI を使わずに CPC だけを記録するスクリプト、センサーの値を読むだけの点検用スクリプト |
| `Python/external/MieScat_Py/` | Mie 散乱の計算ライブラリ |
| `Python/`（直下の `CloudChamber*.py`、`ComputeMatrixA.py`、`Load_*.py` など）、`Python/配布コード/`、`Python/Master/` | 実習で配布するコード（線形回帰による推定の演習用）とその元の版 |
| `Python/2021_data/`、`Python/*.csv` | 過去の実習の観測データと演習用のデータ |
| `Docs/KankyoJissu2026/` | **実習テキスト**（LuaLaTeX）。湿潤空気の熱力学、雲粒の成長、光学、装置、順問題と逆問題（SBI）、観測と解析の課題 |
| `Docs/reference/` | 電子回路工作の参考写真 |
| `Docs/archive/` | 以前のテキスト（2022 年版） |
| `Docs/costs/` | 部品の価格の調べ（論文の Table S1 の出典） |

リポジトリに含まれないもの：観測データ（`data/`、1 日あたり約 18 MB。別にバックアップしてください）、写真と動画、CPC のマニュアル（`CPC3022A/`）、作り直せるキャッシュ（Mie の表 `*.npy`、SBI のシミュレーション結果 `models/*_sims_*.npz`）。

## 使い方

Windows、Miniforge の環境 `cloudchamber`、NI-DAQmx を使います。環境の作り方は `Python/MANUAL.md` の 2 章にあります。

```bat
:: GUI（実機またはシミュレーション）
Python\run_gui.bat

:: スクリプトは必ず環境の中で実行する（python.exe を直接実行すると MKL の読み込みで異常終了する）
cd Python
%USERPROFILE%\miniforge3\Scripts\conda.exe run -n cloudchamber --no-capture-output python check_daq.py Dev2 USB-6210 3
%USERPROFILE%\miniforge3\Scripts\conda.exe run -n cloudchamber --no-capture-output python run_sbi_shape4.py 30000
```

- GUI の「CPC ポート」（初期値 `COM3`）に CPC がつながっていれば、CPC も同じ PC の時計で記録します。
- 実機でバルブが動くのは「自動運転 開始」と、手動運転のボタン操作のときだけです。開始・停止・エラーのときは、バルブを安全な状態（V1 開、V2 閉）にします。

実習テキストのビルド（TeX Live）：

```bat
cd Docs\KankyoJissu2026
latexmk -lualatex -outdir=build KankyoJissu2026.tex
```

## 推定器の版

| 版 | ファイル | 使う Tr の範囲 | モデルの特徴 |
|---|---|---|---|
| 4（現行、論文） | `sbi_shape4.py`、`npe_shape4_*.pt` | 同上 | 前方散乱光を補正した消散効率、記録した圧力の曲線（11 の時刻の圧力降下を入力）、気圧センサーの補正なし（`SBI4_PCORR=1` で戻せる） |
| 3 | `sbi_shape3.py`、`npe_shape3_*.pt` | 下がり始め −0.10〜+0.90 s（2 つ目のこぶまで） | 修正した熱力学、D_v・k_a・表面張力が気温と気圧に依存、気圧センサーの補正、壁からの加熱なし、光路長 0.18 m |
| 2 | `sbi_shape2.py`、`npe_shape2_*.pt` | 同上 | 壁からの加熱と光路長（0.14〜0.20 m）を不確かなパラメータとして扱う。qw が露点計より約 18% 大きい |
| 1 | `sbi_shape.py`、`npe_shape_*.pt` | −0.10〜+0.35 s（最初のこぶまで） | — |

GUI は、学習済みのもののうち最も新しい版を自動で使います。CSV の `estimator` 列に版が記録されます。

## ライセンス

MIT ライセンス（`LICENSE`）。`Python/external/MieScat_Py/` は独自の MIT ライセンスです。
