import pandas as pd
import os

def merge_csv():
    """将某个目录下的CSV文件合格成一个
    """
    # 待处理的目录
    input_path = './save/prediction/'
    result_path = './save/'
    result_name= 'submit.csv'  # 合并后要保存的文件名
    # 获取该目录下所有文件的名字
    file_list = [f'submit_{client_id}.csv' for client_id in range(1, 14)]
    # 读取第一个CSV文件并包含表头
    df = pd.read_csv(input_path + file_list[0], encoding="gbk", header=None)  # 编码默认UTF-8,根据需要需改
    # 将读取的第一个CSV文件写入合并后的文件保存
    df.to_csv(result_path + result_name, encoding="gbk", index=False, header=False)
    # 循环遍历列表中各个CSV文件名，并追加到合并后的文件
    for i in range(1, len(file_list)):
        # 过滤隐藏文件
        if not file_list[i].startswith("."):
            # 根据文件名读取文件
            df = pd.read_csv(input_path + file_list[i], encoding="gbk", header=None)
            # index=True 在最左侧新增索引列；header=True  保留 表头
            df.to_csv(result_path + result_name, encoding="gbk", index=False, header=False, mode='a+')

if __name__ == '__main__':
    merge_csv()