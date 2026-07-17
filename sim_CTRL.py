import os
import netCDF4
import numpy as np
import matplotlib
import subprocess
import matplotlib.pyplot as plt
from config import time_interval_sec
import matplotlib.colors as mcolors
import requests
import json
matplotlib.use('Agg')

"""
#この.pyの狙い
制御ない場合の初期値が欲しい



#やりたいこと

"""


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



def plot_and_save(matrix, title, filename, unit):
    fs = 16
    ticksize = 14
    full_cmap = plt.get_cmap("coolwarm")
    red_cmap = mcolors.ListedColormap(full_cmap(np.linspace(0.5, 1, 256)))  # 赤系の範囲を抽出

    plt.figure(figsize=(10, 8))
    if title == "$\Delta$ RHOT":
        mappable = plt.imshow(matrix, aspect='auto', cmap=red_cmap)
    else:
        mappable = plt.imshow(matrix, aspect='auto', cmap='viridis')
    cbar = plt.colorbar(mappable, label=unit)  # Colorbar を作成
    cbar.set_label(unit, fontsize=fs)    
    cbar.ax.tick_params(labelsize=ticksize)  # 目盛りラベルのフォントサイズを 14 に設定


    plt.title(title, fontsize = fs)
    plt.xlabel('Y', fontsize = fs)
    plt.ylabel('Z', fontsize = fs)
    # 軸の目盛り（ticks）のフォントサイズ変更
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.gca().invert_yaxis()  # 縦軸を反転
    plt.savefig(f"{filename}.pdf", dpi = 1200, bbox_inches = "tight")
    plt.close()

### SCALE-RM関連関数
def prepare_files(pe: int):
    """ファイルの準備と初期化を行う"""
    output_file = f"out.pe######.nc"
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


def CTRL_sim(f):
    MOMY_initial_matrix = np.zeros((97, 40))
    RHOT_initial_matrix = np.zeros((97, 40))
    QV_initial_matrix = np.zeros((97, 40))
    MOMY_min = float("inf")
    RHOT_min = float("inf")
    QV_min = float("inf")
    MOMY_max = - MOMY_min
    RHOT_max = - RHOT_min
    QV_max = - QV_min

    RHOT_diff_vec = np.zeros(97)
    RHOT_diff_matrix = np.zeros((97, 40))

    for pe in range(nofpe):
        init, output = prepare_files(pe)
        with netCDF4.Dataset(init) as src, netCDF4.Dataset(output, "w") as dst:
            # copy global attributes all at once via dictionary
            dst.setncatts(src.__dict__)
            # copy dimensions
            for name, dimension in src.dimensions.items():
                dst.createDimension(
                    name, (len(dimension) if not dimension.isunlimited() else None))
            # copy all file data except for the excluded
            for name, variable in src.variables.items():
                x = dst.createVariable(
                    name, variable.datatype, variable.dimensions)
                # copy variable attributes all at once via dictionary
                dst[name].setncatts(src[name].__dict__)
                print(name)
                if name == "MOMY":
                    var = src[name][:]
                    for Ygrid_i in range(0, 20):
                        for Zgrid_i in range(0, 97):
                            MOMY_initial_matrix[Zgrid_i, Ygrid_i+ pe*20]= var[Ygrid_i, 0, Zgrid_i]  # (y,x,z)
                            if var[Ygrid_i, 0, Zgrid_i] > MOMY_max:
                                MOMY_max = var[Ygrid_i, 0, Zgrid_i]
                            if var[Ygrid_i, 0, Zgrid_i] < MOMY_min:
                                MOMY_min = var[Ygrid_i, 0, Zgrid_i]
                    dst[name][:] = var
                elif name == "RHOT":
                    var = src[name][:]
                    for Ygrid_i in range(0, 20):
                        for Zgrid_i in range(0, 97):
                            RHOT_initial_matrix[Zgrid_i, Ygrid_i+ pe*20]= var[Ygrid_i, 0, Zgrid_i]  # (y,x,z)
                            if var[Ygrid_i, 0, Zgrid_i] > RHOT_max:
                                RHOT_max = var[Ygrid_i, 0, Zgrid_i]
                            if var[Ygrid_i, 0, Zgrid_i] < RHOT_min:
                                RHOT_min = var[Ygrid_i, 0, Zgrid_i]
                            if Ygrid_i == 0 and pe == 0:
                                RHOT_diff_vec[Zgrid_i] =  var[Ygrid_i, 0, Zgrid_i]
                            else:
                                RHOT_diff_matrix[Zgrid_i, Ygrid_i+ pe*20] = var[Ygrid_i, 0, Zgrid_i] - RHOT_diff_vec[Zgrid_i] 
                    dst[name][:] = var
                elif name == "QV":
                    var = src[name][:]
                    for Ygrid_i in range(0, 20):
                        for Zgrid_i in range(0, 97):
                            QV_initial_matrix[Zgrid_i, Ygrid_i+ pe*20]= var[Ygrid_i, 0, Zgrid_i]  # (y,x,z)
                            if var[Ygrid_i, 0, Zgrid_i] >QV_max:
                                QV_max = var[Ygrid_i, 0, Zgrid_i]
                            if var[Ygrid_i, 0, Zgrid_i] < QV_min:
                                QV_min = var[Ygrid_i, 0, Zgrid_i]
                    dst[name][:] = var
                else:
                    dst[name][:] = src[name][:]
                dst[name][:] = src[name][:]
        subprocess.run(["cp", output, init])

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

    f.write(f"\n{MOMY_max=}, {MOMY_min=}")
    f.write(f"\n{RHOT_max=}, {RHOT_min=}")
    f.write(f"\n{QV_max=}, {QV_min=}")
    MOMY_list = MOMY_initial_matrix.tolist()
    RHOT_list = RHOT_initial_matrix.tolist()    
    RHOT_diff_list = RHOT_diff_matrix.tolist()
    QV_list = QV_initial_matrix.tolist()   
    # プロットして保存
    plot_and_save(MOMY_initial_matrix, "MOMY", f"{dirname}/MOMY_initial_Value", '$[kg/m^2/s]$')
    plot_and_save(RHOT_initial_matrix, "RHOT", f"{dirname}/RHOT_initial_Value", '$[kg/m^3]$')
    plot_and_save(QV_initial_matrix, "QV", f"{dirname}/QV_initial_Value", '$[kg/kg]$')
    plot_and_save(RHOT_diff_matrix, "$\Delta$ RHOT", f"{dirname}/RHOT_diff_Value", '$[kg/m^3]$')
    return MOMY_list, RHOT_list, RHOT_diff_list, QV_list

dirname = f"plt_InitialValue/"
os.makedirs(dirname, exist_ok=True)
base_filename = f'MOMY_RHOT_QV'
file_extension = ".txt"
def main():
    # ディレクトリ内のファイルを確認して番号を決定
    counter = 1
    while True:
        filename = f"{base_filename}_{counter}{file_extension}"
        output_file_path = os.path.join(dirname, filename)  
        if not os.path.exists(output_file_path):  
            break
        counter += 1
    with open(output_file_path, 'w')as f:
        MOMY_list, RHOT_list, RHOT_diff_list, QV_list =CTRL_sim(f)

    with open(f'{dirname}{base_filename}/MOMY_data.json', 'w') as f:
        json.dump(MOMY_list, f, indent=4)
    with open(f'{dirname}{base_filename}/RHOT_data.json', 'w') as f:
        json.dump(RHOT_list, f, indent=4)
    with open(f'{dirname}{base_filename}/RHOT_diff_data.json', 'w') as f:
        json.dump(RHOT_diff_list, f, indent=4)
    with open(f'{dirname}{base_filename}/QV_data.json', 'w') as f:
        json.dump(QV_list, f, indent=4)

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