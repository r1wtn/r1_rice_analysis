import pandas as pd
import sys
from pathlib import Path
from shutil import copy2
from random import seed, gauss
from tqdm import tqdm



seed(0)

csv_path = Path(sys.argv[1])
if not csv_path.exists():
    print("file not exists")
    exit()

save_folder = Path(sys.argv[2])
if not save_folder.exists():
    print("save folder does not exist.")
    exit()

df = pd.read_csv(str(csv_path))

print(df)

local_image_path = df["local_image_path"]
print(local_image_path)


# ここから国ごとに分ける処理

groups = df.groupby(by="country_code")

for g in groups:
    code = g[0]
    gdf = g[1]
    new_local_image_path_list = []
    # 国別フォルダの生成
    new_folder = save_folder.joinpath(code, "images")
    if not new_folder.exists():
        new_folder.mkdir(parents=True)
    
    for i, rec in tqdm(gdf.iterrows()):
        p = rec["local_image_path"]
        p = Path(p)
        parts = list(p.parts)
        # yield_raw_datasets 以下の構造がフォルダによって異なるため、親フォルダの場所を動的に変える.
        _i = parts.index("yield_raw_datasets")
        # 現在のフォルダ構成での画像のパス
        src_p = Path("../data_storage/").joinpath("/".join(parts[_i-len(parts):]))
        if not src_p.exists():
            print(src_p)
        # 国別に分割した新規フォルダに保存
        new_p = new_folder.joinpath(f"{i:05d}.jpg")
        new_local_image_path_list.append(new_p)
        # copy2(src_p, new_p)
        
    gdf["local_image_path"] = new_local_image_path_list
    gdf["trainval"] = ["train" if gauss(0, 1) > 0 else "val" for i in range(len(gdf))]

    df_save_path = new_folder.parent.joinpath("data.csv")
    gdf.to_csv(str(df_save_path), index=None)



