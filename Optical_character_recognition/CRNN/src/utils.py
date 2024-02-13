# 运行本脚本进行数据路径索引，生成索引文件
# 包括test.txt、train.txt、val.txt
import os
from tqdm import tqdm


if __name__ == "__main__":

    basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    basedir = basedir.replace("\\","/")
    raw_dir = f'{basedir}/data/raw'
    
    path_list = [f'{raw_dir}/{dir}' for dir in os.listdir(raw_dir)]

    for i,path in enumerate(path_list):
        mode = path.split("/")[-1]
        if mode == "test":
            all_list = [(f'{path_list[i]}/{dir}',dir.split('_')[0]) for dir in os.listdir(path_list[i])]
            for (path,label) in tqdm(all_list):
                with open('./data/labels/test.txt','a',encoding="gbk") as f:
                    f.write(f'{path} {label}\n')
        elif mode == "train":
            all_list = [(f'{path_list[i]}/{dir}',dir.split('_')[0]) for dir in os.listdir(path_list[i])]
            for (path,label) in tqdm(all_list):
                with open('./data/labels/train.txt','a') as f:
                    f.write(f'{path} {label}\n')
        elif mode == "val":
            all_list = [(f'{path_list[i]}/{dir}',dir.split('_')[0]) for dir in os.listdir(path_list[i])]
            for (path,label) in tqdm(all_list):
                with open('./data/labels/val.txt','a') as f:
                    f.write(f'{path} {label}\n')

