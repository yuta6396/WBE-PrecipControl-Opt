import os
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import subprocess
import requests
import contextlib
import json
from skopt import gp_minimize
from skopt.acquisition import gaussian_ei
from skopt.plots import plot_gaussian_process
from skopt.space import Real, Integer
# 時刻を計測するライブラリ
import time
import pytz
from datetime import datetime
from zoneinfo import ZoneInfo

from optimize import random_search
from analysis import *
from make_directory import make_directory
from config import time_interval_sec, bounds
from calc_object_val import calculate_objective_func_val, calculate_objective_sim_val
matplotlib.use('Agg')

"""
BORSのシミュレーション
最適な介入場所(y,z)と介入量MOMYを探す
"""

#### User 設定変数 ##############

input_var = "MOMY" # MOMY, RHOT, QVから選択
max_input = 30#20240830現在ではMOMY=30, RHOT=10, QV=0.1にしている
Opt_purpose = "MinSum" #MinSum, MinMax, MaxSum, MaxMinから選択


max_iter_vec = [15, 15, 20, 50, 50, 50, 50, 50]            #{10, 20, 20, 50]=10, 30, 50, 100と同値
#max_iter_vec = [5,2]       
# initial_design_numdata_vec = [2] #BOのRS回数
# max_iter_vec = [3, 2]  

trial_num =10  #実験試行回数
trial_base = 0

dpi = 75 # 画像の解像度　スクリーンのみなら75以上　印刷用なら300以上
colors6  = ['#4c72b0', '#f28e2b', '#55a868', '#c44e52'] # 論文用の色
###############################
jst = pytz.timezone('Asia/Tokyo')# 日本時間のタイムゾーンを設定
current_time = datetime.now(jst).strftime("%m-%d-%H-%M")

cnt_vec = np.zeros(len(max_iter_vec))
for i in range(len(max_iter_vec)):
    if i == 0:
        cnt_vec[i] = int(max_iter_vec[i])
    else :
        cnt_vec[i] = int(cnt_vec[i-1] + max_iter_vec[i])
"""
gp_minimize で獲得関数を指定: acq_func。
gp_minimize の呼び出しにおける主要なオプションは次の通りです。
"EI": Expected Improvement
"PI": Probability of Improvement
"LCB": Lower Confidence Bound
"gp_hedge": これらの獲得関数をランダムに選択し、探索を行う

EI は、探索と活用のバランスを取りたい場合に多く使用されます。
PI は、最速で最良の解を見つけたい場合に適していますが、早期に探索が止まるリスクがあります。
LCB は、解の探索空間が不確実である場合に有効で、保守的に最適化を進める場合に使用されます
“EIps”
“Expected Improvement per second” を最小化する取得関数です。評価対象の関数は必ず (objective_value, elapsed_time_in_seconds) のタプルを返す必要があります。
これにより、評価コスト（関数呼び出しに要する時間）を考慮しつつ、単位時間あたりの期待改善量を最大化できます。​
“PIps”
“Probability of Improvement per second” を最小化する取得関数です。こちらも (objective_value, elapsed_time_in_seconds) のタプルを返す関数が前提となります。
評価時間を分母に取る点以外は通常の PI（実現確率改善）と同じ概念ですが、時間効率を重視した探索が可能です。​

initial_point_generator
初期探索点をどのようにサンプリングするかを指定します。主な選択肢は以下のとおりです​
"random"： 一様分布によるランダムサンプリング
"sobol"： Sobol シーケンスによる低い不均質系列
"halton"： Halton シーケンスによる低い不均質系列
"hammersly"： Hammersly シーケンスによる低い不均質系列
"lhs"： ラテンハイパーキューブサンプリング
"grid"： グリッド（格子点）方式
"""
acq_func = "gp_hedge"
initial_design_numdata_vec = [10] #BOのRS回数
init_point_generater = "random"

base_dir = f"ICCS_result/JCS_BO/{acq_func}_{init_point_generater}{initial_design_numdata_vec[0]}_{input_var}_seed={trial_base}-{trial_base+trial_num -1}_{current_time}/"



nofpe = 2
fny = 2
fnx = 1
run_time = 20

varname = 'PREC'

init_file = "init_00000101-000000.000.pe######.nc"
org_file = "init_00000101-000000.000.pe######.org.nc"
history_file = "history.pe######.nc"

file_path = os.path.dirname(os.path.abspath(__file__))
gpyoptfile=f"gpyopt.pe######.nc"
orgfile = f'no-control_{str(time_interval_sec)}.pe######.nc'

### SCALE-RM関連関数
def prepare_files(pe: int):
    """ファイルの準備と初期化を行う"""
    output_file = f"out-{input_var}.pe######.nc"
    # input file
    init = init_file.replace('######', str(pe).zfill(6))
    org = org_file.replace('######', str(pe).zfill(6))
    history = history_file.replace('######', str(pe).zfill(6))
    output = output_file.replace('######', str(pe).zfill(6))
    history_path = file_path+'/'+history
    if (os.path.isfile(history_path)):
        subprocess.run(["rm", history])
    subprocess.run(["cp", org, init])  # 初期化

    return init, output

def update_netcdf(init: str, output: str, pe: int, control_input):
    """NetCDFファイルの変数を更新する"""
    Ygrid = control_input[0]
    Zgrid = control_input[1]
    input_MOMY = control_input[2]
    pos = 0 # pe の役割
    if Ygrid > 19:
        Ygrid -=20
        pos = 1

    with netCDF4.Dataset(init) as src, netCDF4.Dataset(output, "w") as dst:
        # グローバル属性のコピー
        dst.setncatts(src.__dict__)
        # 次元のコピー
        for name, dimension in src.dimensions.items():
            dst.createDimension(
                name, (len(dimension) if not dimension.isunlimited() else None))
        # 変数のコピーと更新
        for name, variable in src.variables.items():
            x = dst.createVariable(
                name, variable.datatype, variable.dimensions)
            dst[name].setncatts(src[name].__dict__)
            if name == input_var:
                var = src[name][:]
                if pe == pos:  # pe ==1 =>20~39
                    var[Ygrid, 0, Zgrid] += input_MOMY  
                dst[name][:] = var
            else:
                dst[name][:] = src[name][:]

    # outputをinitにコピー
    subprocess.run(["cp", output, init])
    return init

def sim(control_input):
    """
    制御入力決定後に実際にその入力値でシミュレーションする
    """
    for pe in range(nofpe):
        init, output = prepare_files(pe)
        init = update_netcdf(init, output, pe, control_input)

    subprocess.run(["mpirun", "-n", "2", "./scale-rm", "run_R20kmDX500m.conf"])

    for pe in range(nofpe):
        gpyopt = gpyoptfile.replace('######', str(pe).zfill(6))
        history = history_file.replace('######', str(pe).zfill(6))
        subprocess.run(["cp", history,gpyopt])
    for pe in range(nofpe):  # history処理
        fiy, fix = np.unravel_index(pe, (fny, fnx))
        nc = netCDF4.Dataset(history_file.replace('######', str(pe).zfill(6)))
        onc = netCDF4.Dataset(orgfile.replace('######', str(pe).zfill(6)))
        nt = nc.dimensions['time'].size
        nx = nc.dimensions['x'].size
        ny = nc.dimensions['y'].size
        nz = nc.dimensions['z'].size
        gx1 = nx * fix
        gx2 = nx * (fix + 1)
        gy1 = ny * fiy
        gy2 = ny * (fiy + 1)
        if pe == 0:
            dat = np.zeros((nt, nz, fny*ny, fnx*nx))
            odat = np.zeros((nt, nz, fny*ny, fnx*nx))
        dat[:, 0, gy1:gy2, gx1:gx2] = nc[varname][:]
        odat[:, 0, gy1:gy2, gx1:gx2] = onc[varname][:]

    sum_co=np.zeros(40) #制御後の累積降水量
    sum_no=np.zeros(40) #制御前の累積降水量
    for y_i in range(40):
        for t_j in range(nt):
            if t_j > 0:
                sum_co[y_i] += dat[t_j,0,y_i,0]*time_interval_sec
                sum_no[y_i] += odat[t_j,0,y_i,0]*time_interval_sec
    #print(sum_co-sum_no)
    return sum_co, sum_no

def black_box_function(control_input):
    """
    制御入力値列を入れると、制御結果となる目的関数値を返す
    """
    for pe in range(nofpe):
        init, output = prepare_files(pe)
        init = update_netcdf(init, output, pe, control_input)

    subprocess.run(["mpirun", "-n", "2", "./scale-rm", "run_R20kmDX500m.conf"])

    for pe in range(nofpe):
        gpyopt = gpyoptfile.replace('######', str(pe).zfill(6))
        history = history_file.replace('######', str(pe).zfill(6))
        subprocess.run(["cp", history,gpyopt])
    for pe in range(nofpe):  # history処理
        fiy, fix = np.unravel_index(pe, (fny, fnx))
        nc = netCDF4.Dataset(history_file.replace('######', str(pe).zfill(6)))
        nt = nc.dimensions['time'].size
        nx = nc.dimensions['x'].size
        ny = nc.dimensions['y'].size
        nz = nc.dimensions['z'].size
        gx1 = nx * fix
        gx2 = nx * (fix + 1)
        gy1 = ny * fiy
        gy2 = ny * (fiy + 1)
        if pe == 0:
            dat = np.zeros((nt, nz, fny*ny, fnx*nx))
        dat[:, 0, gy1:gy2, gx1:gx2] = nc[varname][:]

        sum_co=np.zeros(40) #制御後の累積降水量
        for y_i in range(40):
            for t_j in range(nt):
                if t_j > 0: #なぜかt_j=0に　-1*10^30くらいの小さな値が入っているため除外　
                    sum_co[y_i] += dat[t_j,0,y_i,0]*time_interval_sec
    objective_val = calculate_objective_func_val(sum_co, Opt_purpose)

    return objective_val


###実行
def main():
    make_directory(base_dir)
    
    # boundsを定義
    bounds = [Integer(0, 39), Integer(0, 96), Real(-max_input, max_input)]
    
    # 設定情報をJSON形式で保存
    config_data = {
        "input_var": input_var,
        "max_input": max_input,
        "Opt_purpose": Opt_purpose,
        "initial_design_numdata_vec": initial_design_numdata_vec,
        "max_iter_vec": max_iter_vec,
        "trial_num": trial_num,
        "time_interval_sec": time_interval_sec,
        "acq_func": acq_func,
        "init_point_generater": init_point_generater,
        "current_time": current_time,
        "bounds_description": "Y: [0, 39], Z: [0, 96], input_value: [-max_input, max_input]"
    }
    
    config_json_path = os.path.join(base_dir, "config.json")
    with open(config_json_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4, ensure_ascii=False)

    BO_ratio_matrix = np.zeros((len(max_iter_vec), trial_num)) # iterの組み合わせ, 試行回数
    BO_time_matrix = np.zeros((len(max_iter_vec), trial_num)) 
    
    # 結果を格納するためのデータ構造
    optimization_results = {
        "trials": [],
        "summary": {
            "BO_ratio_matrix": [],
            "BO_time_matrix": []
        }
    }

    BO_file = os.path.join(base_dir, "summary", f"BO.txt")
    Sumary_file = os.path.join(base_dir, "summary", f"BO_summary.txt")
    progress_file = os.path.join(base_dir, "progress.txt")


    with open(BO_file, 'w') as f_BO, open(Sumary_file, 'w') as f_s:
        file_handles = {}
        # ExitStackを使用して動的にファイルを開く
        with contextlib.ExitStack() as stack:
            for num in cnt_vec:
                BO_filename = f'BO_{num}.txt'
                # フルパスの生成
                BO_filepath = os.path.join(base_dir, "summary", BO_filename)
                # ファイルを開き、辞書に格納
                file_handles[num] = {
                    'BO': stack.enter_context(open(BO_filepath, 'w', encoding='utf-8'))
                }

            for trial_i in range(trial_num):
                f_s.write(f"\n\n\n{trial_i=}")
                cnt_base = 0
                
                # 各試行の結果を格納するデータ構造
                trial_data = {
                    "trial_number": trial_i,
                    "experiments": []
                }

                for exp_i in range(len(max_iter_vec)):
                    if exp_i > 0:
                        cnt_base  = cnt_vec[exp_i - 1]

                    ###BO
                    random_reset(trial_i+trial_base)

                    start = time.time()  # 現在時刻（処理開始前）を取得
                    # ベイズ最適化の実行
                    if exp_i == 0:
                        result = gp_minimize(
                            func=black_box_function,        # 最小化する関数
                            dimensions=bounds,              # 探索するパラメータの範囲
                            acq_func= acq_func,
                            n_calls=max_iter_vec[exp_i],    # 最適化の反復回数
                            n_initial_points=initial_design_numdata_vec[exp_i],  # 初期探索点の数
                            verbose=True,                   # 最適化の進行状況を表示
                            initial_point_generator = init_point_generater,
                            random_state = trial_i
                        )
                    else:
                        result = gp_minimize(
                            func=black_box_function,        # 最小化する関数
                            dimensions=bounds,              # 探索するパラメータの範囲
                            acq_func=acq_func,
                            n_calls=max_iter_vec[exp_i],    # 最適化の反復回数
                            n_initial_points=0,  # 初期探索点の数
                            verbose=True,                   # 最適化の進行状況を表示
                            initial_point_generator = init_point_generater,
                            random_state = trial_i,
                            x0=initial_x_iters,
                            y0=initial_y_iters
                        )           
                    end = time.time()  # 現在時刻（処理完了後）を取得
                    time_diff = end - start

                    # 最適解の取得
                    min_value = result.fun
                    min_input = result.x
                    print(min_input)
                    initial_x_iters = result.x_iters
                    initial_y_iters = result.func_vals
                    
                    # 実験結果をJSON形式で格納
                    experiment_data = {
                        "experiment_index": exp_i,
                        "num_evaluations": int(cnt_vec[exp_i]),
                        "optimization_result": {
                            "min_value": float(min_value),
                            "min_input": [float(x) for x in min_input],
                            "elapsed_time": float(time_diff),
                            "x_iters": [[float(x) for x in iter_input] for iter_input in result.x_iters],
                            "func_vals": [float(val) for val in result.func_vals]
                        }
                    }
                    
                    f_BO.write(f"\n input\n{result.x_iters}")
                    f_BO.write(f"\n output\n {result.func_vals}")
                    f_BO.write(f"\n最小値:{min_value}")
                    f_BO.write(f"\n入力値:{min_input}")
                    f_BO.write(f"\n経過時間:{time_diff}sec")
                    f_BO.write(f"\nnum_evaluation of BBF = {cnt_vec[exp_i]}")
                    f_s.write(f"\nnum_evaluation of BBF = {cnt_vec[exp_i]}")
                    f_s.write(f"\n最小値:{min_value}")
                    f_s.write(f"\n入力値:{min_input}\n")
                    
                    sum_co, sum_no = sim(min_input)
                    objective_value = calculate_objective_sim_val(sum_co, Opt_purpose)
                    BO_ratio_matrix[exp_i, trial_i] = objective_value
                    BO_time_matrix[exp_i, trial_i] = time_diff
                    
                    # シミュレーション結果を追加
                    experiment_data["simulation_result"] = {
                        "objective_value": float(objective_value),
                        "sum_co": [float(x) for x in sum_co],
                        "sum_no": [float(x) for x in sum_no]
                    }
                    
                    trial_data["experiments"].append(experiment_data)

                    # ファイル操作を行う
                    num = cnt_vec[exp_i]
                    file_handles[num]['BO'].write(f"{min_input}\n")
                
                # 試行の結果をJSONデータに追加
                optimization_results["trials"].append(trial_data)
            
            f_s.write(f"\n全結果:\n{BO_ratio_matrix}\n\n\n")
    
    # 結果行列をJSON形式で保存
    optimization_results["summary"]["BO_ratio_matrix"] = BO_ratio_matrix.tolist()
    optimization_results["summary"]["BO_time_matrix"] = BO_time_matrix.tolist()
    
    # 最終的な結果をJSONファイルに保存
    results_json_path = os.path.join(base_dir, "optimization_results.json")
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(optimization_results, f, indent=4, ensure_ascii=False)
    
    # #シミュレーション結果の可視化
    # filename = f"summary.txt"
    # config_file_path = os.path.join(base_dir, "summary", filename)  
    # f = open(config_file_path, 'w')

    # vizualize_simulation(BO_ratio_matrix, RS_ratio_matrix, BO_time_matrix, RS_time_matrix, max_iter_vec,
    #         f, base_dir, dpi, Alg_vec, colors6, trial_num, cnt_vec)
    # f.close()

def notify_slack(webhook_url, message, channel=None, username=None, icon_emoji=None):
    """
    Slackに通知を送信する関数。
    :param webhook_url: SlackのWebhook URL
    :param message: 送信するメッセージ
    :param channel: メッセージを送信するチャンネル（オプション）
    :param username: メッセージを送信するユーザー名（オプション）
    :param icon_emoji: メッセージに表示する絵文字（オプション）
    """
    payload = {
        "text": message
    }
    # オプションのパラメータを追加
    if channel:
        payload["channel"] = channel
    if username:
        payload["username"] = username
    if icon_emoji:
        payload["icon_emoji"] = icon_emoji
    try:
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()  # エラーがあれば例外を発生させる
        print("Slackへの通知が送信されました。")
    except requests.exceptions.RequestException as e:
        print(f"Slackへの通知に失敗しました: {e}")
def get_script_name():
    return os.path.basename(__file__)
if __name__ == "__main__":
    main()
    webhook_url =os.getenv("SLACK_WEBHOOK_URL") # export SLACK_WEBHOOK_URL="OOOO"したらOK
    # 送信するメッセージを設定
    message = f":チェックマーク_緑: {get_script_name()}の処理が完了しました。"
    notify_slack(webhook_url, message, channel="webhook")
