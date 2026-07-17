import os
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import subprocess
import requests
import contextlib
import json
# 時刻を計測するライブラリ
import time
import pytz
from datetime import datetime
from zoneinfo import ZoneInfo

from config import time_interval_sec, w_max, w_min, crossover_rate, mutation_rate, bounds, types, lower_bound, upper_bound, alpha, tournament_size
from optimize import *
from analysis import *
from make_directory import make_directory
from calc_object_val import calculate_objective_func_val, calculate_objective_sim_val
matplotlib.use('Agg')

"""
PSOGAのシミュレーション
"""

#### User 設定変数 ##############

input_var = "MOMY" # MOMY, RHOT, QVから選択
Alg_vec = ["PSO", "GA"]

Opt_purpose = "MinSum" #MinSum, MinMax, MaxSum, MaxMinから選択

particles_vec = [25, 25, 25, 25, 25, 25]           # 粒子数
iterations_vec = [2, 4, 6, 8, 10, 12]        # 繰り返し回数

# particles_vec = [5, 10]           # 粒子数
# iterations_vec = [3, 3]        # 繰り返し回数

pop_size_vec = particles_vec  # Population size
num_generations_vec = iterations_vec  # Number of generations

# PSO LDWIM
c1 = 2.0
c2 = 2.0

trial_num = 10  # 乱数種の数
trial_base = 0

dpi = 75 # 画像の解像度　スクリーンのみなら75以上　印刷用なら300以上
colors6  = ['#4c72b0', '#f28e2b', '#55a868', '#c44e52'] # 論文用の色
###############################
jst = pytz.timezone('Asia/Tokyo')# 日本時間のタイムゾーンを設定
current_time = datetime.now(jst).strftime("%m-%d-%H-%M")
base_dir = f"ICCS_result/JCS_PSOGA/{Opt_purpose}_{input_var}_seed={trial_base}-{trial_base+trial_num -1}_{current_time}/"
cnt_vec = np.zeros(len(particles_vec))
for i in range(len(particles_vec)):
     cnt_vec[i] = int(particles_vec[i])*int(iterations_vec[i])


nofpe = 2
fny = 2
fnx = 1
run_time = 20


varname = 'PREC'

init_file = "init_00000101-000000.000.pe######.nc"
org_file = "init_00000101-000000.000.pe######.org.nc"
history_file = "history.pe######.nc"

orgfile = f'no-control_{str(time_interval_sec)}.pe######.nc'
file_path = os.path.dirname(os.path.abspath(__file__))
gpyoptfile=f"gpyopt.pe######.nc"


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
    Ygrid = int(round(control_input[0]))  # 第一遺伝子を整数にキャスト
    Zgrid = int(round(control_input[1]))  # 第二遺伝子を整数にキャスト
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
            control_dat = np.zeros((nt, nz, fny*ny, fnx*nx))
            no_control_odat = np.zeros((nt, nz, fny*ny, fnx*nx)) 

        dat[:, 0, gy1:gy2, gx1:gx2] = nc[varname][:]
        odat[:, 0, gy1:gy2, gx1:gx2] = onc[varname][:]
        
    sum_co=np.zeros(40) #制御後の累積降水量
    sum_no=np.zeros(40) #制御前の累積降水量
    for y_i in range(40):
        for t_j in range(nt):
            if t_j > 0:
                sum_co[y_i] += dat[t_j,0,y_i,0]*time_interval_sec
                sum_no[y_i] += odat[t_j,0,y_i,0]*time_interval_sec
    return sum_co, sum_no

def black_box_function(control_input):
    """
    制御入力値列を入れると、制御結果となる目的関数値を返す
    """
    print(control_input)
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

    # 設定情報をJSON形式で保存
    config_data = {
        "input_var": input_var,
        "Alg_vec": Alg_vec,
        "Opt_purpose": Opt_purpose,
        "particles_vec": particles_vec,
        "iterations_vec": iterations_vec,
        "pop_size_vec": pop_size_vec,
        "num_generations_vec": num_generations_vec,
        "cnt_vec": cnt_vec.tolist(),
        "trial_num": trial_num,
        "trial_base": trial_base,
        "dpi": dpi,
        "time_interval_sec": time_interval_sec,
        "current_time": current_time,
        "PSO_parameters": {
            "w_max": w_max,
            "w_min": w_min,
            "c1": c1,
            "c2": c2
        },
        "GA_parameters": {
            "crossover_rate": crossover_rate,
            "mutation_rate": mutation_rate,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "alpha": alpha,
            "tournament_size": tournament_size
        },
        "bounds_description": "Y: [0, 39], Z: [0, 96], input_value: variable range"
    }
    
    config_json_path = os.path.join(base_dir, "config.json")
    with open(config_json_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4, ensure_ascii=False)

    PSO_ratio_matrix = np.zeros((len(particles_vec), trial_num))
    GA_ratio_matrix = np.zeros((len(particles_vec), trial_num))
    PSO_time_matrix = np.zeros((len(particles_vec), trial_num))
    GA_time_matrix = np.zeros((len(particles_vec), trial_num))
    
    # 結果を格納するためのデータ構造
    optimization_results = {
        "trials": [],
        "summary": {
            "PSO_ratio_matrix": [],
            "GA_ratio_matrix": [],
            "PSO_time_matrix": [],
            "GA_time_matrix": []
        }
    }

    PSO_file = os.path.join(base_dir, "summary", f"{Alg_vec[0]}.txt")
    GA_file = os.path.join(base_dir, "summary", f"{Alg_vec[1]}.txt")
    progress_file = os.path.join(base_dir, "progress.txt")

    with open(PSO_file, 'w') as f_PSO, open(GA_file, 'w') as f_GA,  open(progress_file, 'w') as f_progress:
        file_handles = {}
        # ExitStackを使用して動的にファイルを開く
        with contextlib.ExitStack() as stack:
            for num in cnt_vec:
                PSO_filename = f'PSO_{num}.txt'
                GA_filename = f'GA_{num}.txt'
                # フルパスの生成
                PSO_filepath = os.path.join(base_dir, "summary", PSO_filename)
                GA_filepath = os.path.join(base_dir, "summary", GA_filename)
                # ファイルを開き、辞書に格納
                file_handles[num] = {
                    'PSO': stack.enter_context(open(PSO_filepath, 'w', encoding='utf-8')),
                    'GA': stack.enter_context(open(GA_filepath, 'w', encoding='utf-8'))
                }
            for trial_i in range(trial_num):
                f_progress.write(f"\n\n{trial_i=}\n")
                cnt_base = 0
                
                # 各試行の結果を格納するデータ構造
                trial_data = {
                    "trial_number": trial_i,
                    "experiments": []
                }
                
                for exp_i in range(len(particles_vec)):
                    f_progress.write(f"{exp_i=},  ")
                    if exp_i > 0:
                        cnt_base  = cnt_vec[exp_i - 1]


                    ###PSO
                    random_reset(trial_i+trial_base)

                    start = time.time()  # 現在時刻（処理開始前）を取得
                    best_position, result_value  = PSO(black_box_function, bounds,types,  particles_vec[exp_i], iterations_vec[exp_i], f_PSO)
                    end = time.time()  # 現在時刻（処理完了後）を取得
                    time_diff = end - start
                    
                    sum_co, sum_no = sim(best_position)
                    objective_value = calculate_objective_sim_val(sum_co, Opt_purpose)
                    PSO_ratio_matrix[exp_i, trial_i] = objective_value
                    PSO_time_matrix[exp_i, trial_i] = time_diff
                    
                    # PSO実験結果をJSON形式で格納
                    pso_experiment_data = {
                        "algorithm": "PSO",
                        "experiment_index": exp_i,
                        "num_evaluations": int(cnt_vec[exp_i]),
                        "optimization_result": {
                            "min_value": float(result_value[iterations_vec[exp_i]-1]),
                            "min_input": [float(x) for x in best_position],
                            "elapsed_time": float(time_diff),
                            "particles": particles_vec[exp_i],
                            "iterations": iterations_vec[exp_i]
                        },
                        "simulation_result": {
                            "objective_value": float(objective_value),
                            "sum_co": [float(x) for x in sum_co]
                        }
                    }
                    
                    f_PSO.write(f"\n最小値:{result_value[iterations_vec[exp_i]-1]}")
                    f_PSO.write(f"\n入力値:{best_position}")
                    f_PSO.write(f"\n経過時間:{time_diff}sec")
                    f_PSO.write(f"\nnum_evaluation of BBF = {cnt_vec[exp_i]}")

                    ###GA
                    random_reset(trial_i+trial_base)
                    # パラメータの設定
                    start = time.time()  # 現在時刻（処理開始前）を取得
                    # Run GA with the black_box_function as the fitness function
                    best_fitness, best_individual = genetic_algorithm(black_box_function,
                        pop_size_vec[exp_i], num_generations_vec[exp_i],
                        crossover_rate, mutation_rate, lower_bound, upper_bound,
                        alpha, tournament_size, types, f_GA)
                    end = time.time()  # 現在時刻（処理完了後）を取得
                    time_diff = end - start

                    sum_co, sum_no = sim(best_individual)
                    objective_value = calculate_objective_sim_val(sum_co, Opt_purpose)
                    GA_ratio_matrix[exp_i, trial_i] = objective_value
                    GA_time_matrix[exp_i, trial_i] = time_diff
                    
                    # GA実験結果をJSON形式で格納
                    ga_experiment_data = {
                        "algorithm": "GA",
                        "experiment_index": exp_i,
                        "num_evaluations": int(cnt_vec[exp_i]),
                        "optimization_result": {
                            "min_value": float(best_fitness),
                            "min_input": [float(x) for x in best_individual],
                            "elapsed_time": float(time_diff),
                            "population_size": pop_size_vec[exp_i],
                            "generations": num_generations_vec[exp_i]
                        },
                        "simulation_result": {
                            "objective_value": float(objective_value),
                            "sum_co": [float(x) for x in sum_co],
                            "sum_no": [float(x) for x in sum_no]
                        }
                    }
                    
                    trial_data["experiments"].extend([pso_experiment_data, ga_experiment_data])

                    f_GA.write(f"\n最小値:{best_fitness}")
                    f_GA.write(f"\n入力値:{best_individual}")
                    f_GA.write(f"\n経過時間:{time_diff}sec")
                    f_GA.write(f"\nnum_evaluation of BBF = {cnt_vec[exp_i]}")

                    num = cnt_vec[exp_i]
                    file_handles[num]['PSO'].write(f"{best_position}\n")
                    file_handles[num]['GA'].write(f"{best_individual}\n")
                
                # 試行の結果をJSONデータに追加
                optimization_results["trials"].append(trial_data)

        # 例: PSO_ratio_matrix をCSVファイルとして保存
        np.savetxt(os.path.join(base_dir, "summary", "PSO_ratio_matrix.csv"), PSO_ratio_matrix, delimiter=",")
        np.savetxt(os.path.join(base_dir, "summary", "GA_ratio_matrix.csv"), GA_ratio_matrix, delimiter=",")
        np.savetxt(os.path.join(base_dir, "summary", "PSO_time_matrix.csv"), PSO_time_matrix, delimiter=",")
        np.savetxt(os.path.join(base_dir, "summary", "GA_time_matrix.csv"), GA_time_matrix, delimiter=",")

    # 結果行列をJSON形式で保存
    optimization_results["summary"]["PSO_ratio_matrix"] = PSO_ratio_matrix.tolist()
    optimization_results["summary"]["GA_ratio_matrix"] = GA_ratio_matrix.tolist()
    optimization_results["summary"]["PSO_time_matrix"] = PSO_time_matrix.tolist()
    optimization_results["summary"]["GA_time_matrix"] = GA_time_matrix.tolist()
    
    # 最終的な結果をJSONファイルに保存
    results_json_path = os.path.join(base_dir, "optimization_results.json")
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(optimization_results, f, indent=4, ensure_ascii=False)

    #シミュレーション結果の可視化
    filename = f"summary.txt"
    config_file_path = os.path.join(base_dir, "summary", filename)  
    f = open(config_file_path, 'w')

    vizualize_simulation(PSO_ratio_matrix, GA_ratio_matrix, PSO_time_matrix, GA_time_matrix, particles_vec,
            f, base_dir, dpi, Alg_vec, colors6, trial_num, cnt_vec)

    f.close()

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