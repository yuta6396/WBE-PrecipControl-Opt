import os
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import subprocess
import requests
import contextlib
from skopt import gp_minimize
from skopt.acquisition import gaussian_ei
from skopt.plots import plot_gaussian_process
from skopt.space import Real, Integer
# 時刻を計測するライブラリ
import time
import pytz
from datetime import datetime
from zoneinfo import ZoneInfo

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

random_iter_vec = [15]            #{10, 20, 20, 50]=10, 30, 50, 100と同値
# random_iter_vec = [3, 2]  

trial_num = 1  #箱ひげ図作成時の繰り返し回数
trial_base = 0

dpi = 75 # 画像の解像度　スクリーンのみなら75以上　印刷用なら300以上
colors6  = ['#4c72b0', '#f28e2b', '#55a868', '#c44e52'] # 論文用の色
###############################
jst = pytz.timezone('Asia/Tokyo')# 日本時間のタイムゾーンを設定
current_time = datetime.now(jst).strftime("%m-%d-%H-%M")
base_dir = f"ICCS_result/RS/{Opt_purpose}_{input_var}_seed={trial_base}-{trial_base+trial_num -1}_{current_time}/"


cnt_vec = np.zeros(len(random_iter_vec))
for i in range(len(random_iter_vec)):
    if i == 0:
        cnt_vec[i] = int(random_iter_vec[i])
    else :
        cnt_vec[i] = int(cnt_vec[i-1] + random_iter_vec[i])
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
"""




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

def random_search(objective_function, bounds, f_in, f_RS, RS_ratio_matrix, trial_i):
    # 以前の最良のスコアとパラメータを初期化
    for i in range(len(random_iter_vec)):
        input_history=[]
        if i==0:
            best_score = float('inf')
            best_params = None
        for _ in range(random_iter_vec[i]):
            candidate = []
            for b in bounds:
                if b[0] == 'int':
                    # 整数値を生成
                    value = np.random.randint(b[1], b[2] + 1)  # 上限は含まないため +1
                elif b[0] == 'float':
                    # 実数値を生成
                    value = np.random.uniform(b[1], b[2])
                else:
                    raise ValueError(f"Unsupported type: {b[0]}")
                candidate.append(value)
            input_history.append(candidate)
            score = objective_function(candidate)
            if score < best_score:
                best_score = score
                best_params = candidate

        sum_co, sum_no = sim(best_params)
        sum_RS_MOMY = sum_co
        RS_ratio_matrix[i, trial_i] =  calculate_objective_sim_val(sum_co, Opt_purpose)
        f_in.write(f"BBF={cnt_vec[i]}\n input_history=\n{input_history}\n\n\n")
        f_RS.write(f"\nnum_evaluation of BBF = {cnt_vec[i]}") 
        f_RS.write(f"\n最小値:{best_score}")
        f_RS.write(f"\n入力値:{best_params}\n\n")
    if trial_i == trial_num-1:
        f_RS.write(f"\n全結果:\n{RS_ratio_matrix}\n\n\n")
    return 
###実行
def main():
    print(Opt_purpose)
    make_directory(base_dir)
    
    filename = f"config.txt"
    config_file_path = os.path.join(base_dir, filename)  # 修正ポイント
    ##設定メモ##
    with open(config_file_path, 'w') as f:
        f.write(f"input_var ={input_var}")
        f.write(f"\nmax_input ={max_input}")
        f.write(f"\nOpt_purpose ={Opt_purpose}")
        f.write(f"\nrandom_iter_vec = {random_iter_vec}")
        f.write(f"\ntrial_num = {trial_num}")
        f.write(f"\n{time_interval_sec=}")

    RS_ratio_matrix = np.zeros((len(random_iter_vec), trial_num))

    RS_file = os.path.join(base_dir, "summary", "RS.txt")
    RS_input_file = os.path.join(base_dir, "summary", "RS_input_data.txt")
    progress_file = os.path.join(base_dir, "progress.txt")


    with open(RS_file, 'w') as f_RS, open(RS_input_file, 'w') as f_in:
        for trial_i in range(trial_num):
            ###RS
            random_reset(trial_i+trial_base)
            # パラメータの設定
            # bounds_MOMY = [(-max_input, max_input)]*num_input_grid  # 探索範囲
            bounds = [
                        ('int', 0, 39),
                        ('int', 0, 96),
                        ('float', -max_input, max_input)
                    ]

            random_search(black_box_function, bounds,  f_in, f_RS, RS_ratio_matrix,trial_i)

    #シミュレーション結果の可視化
    filename = f"summary.txt"
    config_file_path = os.path.join(base_dir, "summary", filename)  
    f = open(config_file_path, 'w')

    # vizualize_simulation(BO_ratio_matrix, RS_ratio_matrix, BO_time_matrix, RS_time_matrix, random_iter_vec,
    #         f, base_dir, dpi, Alg_vec, colors6, trial_num, cnt_vec)
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