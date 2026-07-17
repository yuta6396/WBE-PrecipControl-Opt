# WBE-PrecipControl-Opt — Warm Bubble Experiment（降水制御の最適化実験）

SCALE-RM の理想化実験「Warm Bubble Experiment（暖気塊実験）」を対象に、
**局所的な介入（制御入力）によって累積降水量を最小化する最適な介入場所・介入量を探索する**
一連の実験コード置き場です。

介入は「どの Y-grid・Z-grid に、どれだけの運動量（MOMY 等）を加えるか」という
3 次元の制御入力 `(Ygrid, Zgrid, input_value)` として与え、
ブラックボックス最適化（BO / RS / PSO / GA / GS）で最適解を探します。

> 元々は ICCS2025 発表に向けて始めたコード（旧 `WBE-ICCS`）を整理し直したものです。現在は
> **ICCS 系** と **JCS 系** の 2 系統の実験が同居しています（下記参照）。
>
> **注意（このリポジトリに含まれないもの）**: SCALE-RM の実行バイナリ
> （`scale-rm` / `scale-rm_init` / `sno`）はサイズが大きいため Git 管理対象外です。
> 実行するには、各自の SCALE-RM ビルドからこれらをこのディレクトリへ
> コピー（またはシンボリックリンク）してください。詳細は「5. 実行方法」を参照。

---

## 1. 実験系列（2 系統）

このリポジトリには目的の異なる 2 つの実験系列があり、ファイル名の接頭辞で区別します。

| 系列 | ファイル接頭辞 | 目的 | 出力先 |
|------|----------------|------|--------|
| **ICCS 系** | `sim_ICCS_*.py` | ICCS2025 向けの基本実験。各最適化手法の動作確認・少数試行での比較が中心。 | `ICCS_result/<手法>/` |
| **JCS 系** | `sim_JCS_*.py` | **ハイパーパラメータを振って**手法性能を比較する本番実験。試行回数が多く（`trial_num=10` など）、設定・結果を JSON でも保存。 | `ICCS_result/JCS_<手法>/` |

### ICCS 系と JCS 系の主な違い
- **試行回数 / 反復回数**: JCS 系は `trial_num` や `max_iter_vec`（BO）/ `particles_vec`・`iterations_vec`（PSOGA）を大きく取り、統計的な比較ができるようにしている。
- **結果の保存形式**: JCS 系は `config.json` と `optimization_results.json` に設定・全試行結果を機械可読な形で保存する（ICCS 系は主にテキスト出力）。
- **共通部分**: SCALE-RM の呼び出し・NetCDF 入出力・目的関数の計算（`calc_object_val.py`）・最適化アルゴリズム（`optimize.py`）は両系列で共通。

---

## 2. 最適化手法（サフィックスの意味）

`sim_<系列>_<手法>.py` の `<手法>` は次を表します。

| サフィックス | 手法 | 概要 |
|--------------|------|------|
| `BO` | ベイズ最適化 (Bayesian Optimization) | `scikit-optimize` の `gp_minimize` を使用。獲得関数 `acq_func`（`EI`/`PI`/`LCB`/`gp_hedge`）を切り替え可能。 |
| `RS` | ランダムサーチ (Random Search) | ベースライン。`optimize.random_search` を使用。 |
| `PSOGA` | PSO / GA | 粒子群最適化と遺伝的アルゴリズム。`optimize.py` の `PSO` / `genetic_algorithm` を使用。 |
| `GS` | グリッドサーチ (Grid Search) | 全格子点を総当たり評価（ICCS 系のみ）。`grid_search_3d_heatmap.html` を生成。 |
| `1run` | 単発実行 | 指定した 1 つの制御入力でのシミュレーション・可視化（ICCS 系のみ）。 |
| `CTRL` | 制御なし基準の作成 | 介入しない場合の基準降水量（`no-control_*.nc`）を作るためのスクリプト。 |

`sim_ICCS_main.py` は複数スクリプトを順番に実行するためのランナー、
`sim_6inputs_3var_BO.py` / `sim_test_BO.py` は派生・検証用の実験スクリプトです。

---

## 3. ディレクトリ / ファイル構成

```
WBE-PrecipControl-Opt/
├── README.md                     # 本ファイル
├── .gitignore
│
├── ── 実験スクリプト（ICCS 系）──
│   ├── sim_ICCS_BO.py            # ベイズ最適化
│   ├── sim_ICCS_RS.py            # ランダムサーチ
│   ├── sim_ICCS_PSOGA.py         # PSO / GA
│   ├── sim_ICCS_GS.py            # グリッドサーチ
│   ├── sim_ICCS_1run.py          # 単発実行・可視化
│   └── sim_ICCS_main.py          # 複数スクリプトの一括ランナー
│
├── ── 実験スクリプト（JCS 系）──
│   ├── sim_JCS_BO.py             # ベイズ最適化（ハイパラ比較・JSON 保存）
│   ├── sim_JCS_RS.py             # ランダムサーチ
│   └── sim_JCS_PSOGA.py          # PSO / GA
│
├── ── その他の実験スクリプト ──
│   ├── sim_CTRL.py               # 制御なし基準（no-control）の作成
│   ├── sim_6inputs_3var_BO.py    # 派生実験（多入力）
│   └── sim_test_BO.py            # 検証用
│
├── ── 共通モジュール ──
│   ├── config.py                 # PSO/GA パラメータ・探索範囲 bounds・時間間隔など
│   ├── calc_object_val.py        # 目的関数（累積降水量の削減率）の計算
│   ├── optimize.py               # RS / PSO / GA の実装
│   ├── analysis.py               # 結果集計・グラフ・アニメーション作成
│   ├── make_directory.py         # 結果出力用ディレクトリ構造の作成
│   ├── visualize_input.py        # 初期値の可視化（ノートブック的スクリプト）
│   └── sample_anime.py           # history からのアニメーション作成サンプル
│
├── ── SCALE-RM 実行環境 ──
│   ├── (scale-rm, scale-rm_init, sno)  # 実行バイナリ。リポジトリ非同梱。各自でコピー/リンクする
│   ├── run_R20kmDX500m.conf      # メインの実験設定（半径20km, DX500m, 2D）
│   ├── run_R20kmDX500m_1200sec.conf
│   └── sno_R20kmDX500m.conf
│
├── ── 入力・基準データ（NetCDF）──
│   ├── init_*.pe*.org.nc         # 制御なしのオリジナル初期値（★変更しないこと）
│   ├── init_*.pe*.nc             # 実行時に org からコピーされる作業用初期値（自動生成）
│   └── no-control_*.pe*.nc       # 制御なし基準ラン（目的関数の分母・比較対象）
│
└── ── 出力 ──
    ├── ICCS_result/              # 実験結果（.gitignore 済み。手法・系列ごとにサブディレクトリ）
    └── plt_InitialValue/         # 初期値プロット
```

> **注意**: `init_*.pe*.nc`（`.org` なし）は実行のたびに `init_*.pe*.org.nc` からコピー・上書きされる作業ファイルです。オリジナルは `*.org.nc` の方なので、そちらは絶対に書き換えないでください。

---

## 4. 実験のしくみ（1 評価の流れ）

各最適化スクリプトの「1 回の目的関数評価」は、`black_box_function(control_input)` で次を行います。

1. `init_*.pe*.org.nc` を `init_*.pe*.nc` にコピーして初期化（`prepare_files`）。
2. 制御入力 `(Ygrid, Zgrid, input_value)` を初期値 NetCDF の対象変数（`input_var`, 既定は `MOMY`）に加算（`update_netcdf`）。
3. `mpirun -n 2 ./scale-rm run_R20kmDX500m.conf` で SCALE-RM を実行（2 プロセス並列、2D 実験）。
4. `history.pe*.nc` から降水量 `PREC` を読み、各 Y-grid の累積降水量 `sum_co` を算出。
5. `calc_object_val.calculate_objective_func_val(sum_co, Opt_purpose)` で目的関数値（累積降水量削減率など）を計算し、最小化する。

`Opt_purpose` は `MinSum`（合計最小化）/ `MinMax`（最大地点の最小化）/ `MaxSum` / `MaxMax` から選択します。

---

## 5. 実行方法

### 事前準備（初回のみ）

clone しただけでは実行できません。**SCALE-RM の実行バイナリはリポジトリに含まれていない**ため、
自分の SCALE-RM ビルドからリンク（またはコピー）する必要があります。

#### シミュレーションに必要なファイル一覧

| ファイル | 役割 | このリポジトリに | 用意方法 |
|----------|------|------------------|----------|
| `scale-rm` | SCALE-RM 本体（**実行に必須**。全 `sim_*.py` が `./scale-rm` を呼ぶ） | ✗ 非同梱 | ビルド先から `ln -s`（下記） |
| `run_R20kmDX500m.conf` | 実験設定（半径20km・DX500m・2D、`TIME_DURATION=3600s`） | ✓ 同梱 | そのまま使用 |
| `run_R20kmDX500m_1200sec.conf` | 短時間版設定（`sim_test_BO` / `sim_6inputs_3var_BO` 用） | ✓ 同梱 | そのまま使用 |
| `init_00000101-000000.000.pe000{0,1}.org.nc` | **オリジナル初期値**（介入なしの初期状態） | ✓ 同梱 | そのまま使用（書き換え禁止） |
| `init_00000101-000000.000.pe000{0,1}.nc` | 作業用初期値（`.conf` の `RESTART_IN_BASENAME`） | ✗ 自動生成 | 各評価の冒頭で `.org.nc` から自動コピー |
| `no-control_{300,3600}.pe000{0,1}.nc` | 制御なし基準ラン（目的関数の比較対象） | ✓ 同梱 | そのまま使用 |
| `scale-rm_init` / `sno` | 初期値の再生成 / NetCDF 後処理（**通常の最適化実行には不要**） | ✗ 非同梱 | 必要な場合のみ `ln -s` |

> ポイント: `scale-rm` バイナリさえ用意すれば最適化実験は動きます。作業用 `init_*.pe*.nc` は
> `prepare_files()` が毎評価ごとに `.org.nc` からコピーして作るので、手動で置く必要はありません。

#### 1. SCALE-RM 実行バイナリのリンク

ビルド済み SCALE-RM の `bin/` から、このディレクトリへシンボリックリンクを張ります。

```bash
cd WBE-PrecipControl-Opt

# 例) この環境の SCALE-RM ビルド（/home/yuta/scale-5.5.1/bin）を絶対パスでリンク
ln -s /home/yuta/scale-5.5.1/bin/scale-rm       scale-rm
# 初期値を作り直したい / 後処理をしたい場合のみ（通常の実行には不要）
ln -s /home/yuta/scale-5.5.1/bin/scale-rm_init  scale-rm_init
ln -s /home/yuta/scale-5.5.1/bin/sno            sno
```

別の場所に置いた場合は、そのパスに読み替えてください。リンクが正しいかは以下で確認できます。
```bash
ls -l scale-rm            # -> ... scale-rm -> /.../bin/scale-rm
./scale-rm --version 2>/dev/null || echo "リンク先を確認してください"
```

#### 2. 実行環境

- MPI（`mpirun`）が利用可能なこと（`run_R20kmDX500m.conf` は `PRC_NUM_Y=2` の 2 プロセス構成）。
- Python 3.11 系。主な依存: `numpy`, `netCDF4`, `matplotlib`, `scikit-optimize (skopt)`, `pytz`, `requests`, `pandas`, `plotly`。

```bash
# 例: 依存パッケージのインストール
pip install numpy netCDF4 matplotlib scikit-optimize pytz requests pandas plotly
```

### 実行例
```bash
cd WBE-PrecipControl-Opt

# ICCS 系: ベイズ最適化
python sim_ICCS_BO.py

# JCS 系: ハイパーパラメータを振ったベイズ最適化（本番）
python sim_JCS_BO.py

# 制御なし基準（no-control_*.nc）を作り直したいとき
python sim_CTRL.py
```

実験条件は各スクリプト冒頭の「`#### User 設定変数 ####`」ブロックで調整します
（`input_var`, `max_input`, `Opt_purpose`, `max_iter_vec`, `trial_num`, `acq_func` など）。

### Slack 通知（任意）
各スクリプトは完了時に Slack へ通知できます。使う場合のみ環境変数を設定してください。
```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/XXXX"
```

---

## 6. 出力の見かた

結果は `ICCS_result/<系列・手法>/<設定>_<日時>/` 以下に作られます（`make_directory.py`）。

```
<結果ディレクトリ>/
├── config.txt / config.json      # 実行時の設定
├── optimization_results.json     # 全試行の入力・出力（JCS 系）
├── progress.txt
└── summary/                      # 最良入力・目的関数値のまとめ、比較用テキスト
    ├── BO.txt, BO_summary.txt, BO_<n>.txt ...
```

`ICCS_result/` は容量が大きくなるため Git 管理対象外（`.gitignore` 済み）です。

---

## 7. 設定パラメータの要点（`config.py`）

- `time_interval_sec`: 出力の時間間隔（秒）。`.conf` の `FILE_HISTORY_DEFAULT_TINTERVAL` と一致させること。
- `bounds` / `lower_bound` / `upper_bound`: 探索範囲 `[(Ygrid 0–39), (Zgrid 0–96), (input -30–30)]`。
- PSO: `w_max`, `w_min`（慣性重み）。GA: `crossover_rate`, `mutation_rate`, `alpha`(BLX-α), `tournament_size`。

---

## 8. リポジトリ運用メモ

- 旧 `WBE-ICCS` からコード・設定・オリジナル入力データのみを移植して作成した新リポジトリ。
- 次のものは `.gitignore` 済み（コミットしない）:
  - SCALE-RM の出力（`out*.nc`, `gpyopt*.nc`, `history*.nc`, `LOG*`）
  - 実行時の作業用 `init_*.pe*.nc`
  - SCALE-RM 実行バイナリ（`scale-rm`, `scale-rm_init`, `sno`）
  - 結果ディレクトリ（`ICCS_result/`, `test_result/`, `plt_InitialValue/`）・`__pycache__/`
- コミット対象は「スクリプト・設定・共通モジュール・オリジナル入力データ（`*.org.nc`, `no-control_*.nc`）」を基本とする。
