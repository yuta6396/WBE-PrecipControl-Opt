import os
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import subprocess
import requests
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

import pandas as pd
import plotly.graph_objs as go
import plotly.graph_objs as go
matplotlib.use('Agg')

"""
GSシミュレーション
"""

#### User 設定変数 ##############

input_var = "MOMY" # MOMY, RHOT, QVから選択
Opt_purpose = "MinSum" #MinSum, MinMax, MaxSum, MaxMinから選択 #ICCSはMinSum

dpi = 75 # 画像の解像度　スクリーンのみなら75以上　印刷用なら300以上
colors6  = ['#4c72b0', '#f28e2b', '#55a868', '#c44e52'] # 論文用の色
###############################
jst = pytz.timezone('Asia/Tokyo')# 日本時間のタイムゾーンを設定
current_time = datetime.now(jst).strftime("%m-%d-%H-%M")
base_dir = f"ICCS_result/1run/{Opt_purpose}_{input_var}{upper_bound[2]}_{current_time}/"


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
            # print(name)
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

    objective_val = calculate_objective_sim_val(sum_co, Opt_purpose)
    return objective_val, sum_co, sum_no

def plot_bar_graph(sum_co, sum_no):
    """
    sum_co, sum_no: それぞれ長さ40のnumpy配列
    """
    indices = np.arange(len(sum_co))  # インデックス0～39

    plt.figure(figsize=(12, 5))
    
    # sum_co の棒グラフ
    plt.subplot(1, 2, 1)
    plt.bar(indices, sum_co, color='skyblue')
    plt.xlabel("Y")
    plt.ylabel("PREC [mm/h]")
    plt.title("Bar Graph for TEST")
    
    # sum_no の棒グラフ
    plt.subplot(1, 2, 2)
    plt.bar(indices, sum_no, color='salmon')
    plt.xlabel("Y")
    plt.ylabel("PREC [mm/h]")
    plt.title("Bar Graph for CTRL")
    
    plt.tight_layout()
    pdf_path = os.path.join(base_dir, "bar_graph.pdf")
    # PDF として保存
    plt.savefig(pdf_path, format="pdf")
    print(f"Figure saved as {pdf_path}")

###実行
def main():
    os.makedirs(base_dir, exist_ok=True)
    output_file_path = os.path.join(base_dir, f'summary.txt')
    with open(output_file_path, 'w') as f:
        # [0, 39], [0, 96], (-30, 30)
        objective_val, sum_co, sum_no = sim( [17, 6, 30])
        # 300回の時の最適解　BO [20, 5, -24.92778677955823] RS[15, 3, 25.09999404257084] PSO [17, 6, 30] GA[13,14, 29.67936212]
        f.write(f"{sum_co=}\n\n")
        f.write(f"{sum_no=}")
        plot_bar_graph(sum_co, sum_no)
        print(objective_val)


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