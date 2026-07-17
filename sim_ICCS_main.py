import subprocess
import json
import sys
import os



def run_script(script_name):
    try:
        # コマンドライン引数として config_file を渡す
        result = subprocess.run(
            ['python', script_name],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Output of {script_name}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}: {e.stderr}", file=sys.stderr)

def main():
    # 実行したいスクリプトのリスト
    scripts = [
        'sim_ICCS_BORS.py', 'sim_ICCS_PSOGA.py'
        # 他のスクリプトをここに追加
        # 'another_script.py',
        # 'yet_another_script.py',
    ]

    # スクリプトが存在するか確認
    for script in scripts:
        if not os.path.isfile(script):
            print(f"Script {script} does not exist.", file=sys.stderr)
            return

    # 各スクリプトを順番に実行
    for script in scripts:
        print(f"Running {script}...")
        run_script(script)
        print(f"Finished running {script}.\n")

if __name__ == "__main__":
    main()
