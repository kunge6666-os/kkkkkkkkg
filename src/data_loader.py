"""
数据加载模块
负责数据读取、抽样和基础验证
"""
import pandas as pd
import numpy as np
import time
from config import RAW_DATA_PATH


class DataLoader:
    def __init__(self):
        self.raw_data_path = RAW_DATA_PATH


    def load_random(self,sample_size=5000):
        start_time=time.time()
        print("正在读取数据...")
        with open(self.raw_data_path, "r", encoding="utf-8") as f:
            total_raws = sum(1 for _ in f) - 1  # 减去标题行
        print(f"记录总行数: {total_raws:,}")
        skip_raws = np.random.choice(range(1, total_raws + 1),
                                     size=total_raws - sample_size, replace=False)
        df = pd.read_csv(self.raw_data_path,
                         names=['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp'],
                         skiprows=skip_raws)
        print(f"数据加载完成:{len(df):,}条记录")
        print(f"加载时间:{time.time()-start_time:.2f}秒")
        return df


    def load_with_sampling(self,sample_size=500000,random_state=42):
        """
    Args:
        sample_size:抽样大小
        random_state:随机种子
        """
        print("大数据集抽样加载")
        start_time = time.time()

        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.raw_data_path}")
        try:
            #计算总行数
            print("正在计算数据量...")
            with open(self.raw_data_path, "r", encoding="utf-8") as f:
                total_raws= sum(1 for _ in f)-1 #减去标题行
            print(f"记录总行数: {total_raws:,}")
            if total_raws < sample_size:
                print("样本大于总行数，使用全部数据")
                df=pd.read_csv(self.raw_data_path,
                               names=['user_id','item_id','category_id','behavior_type','timestamp'])
            else:
                #随机抽样
                skip_raws = np.random.choice(range(1,total_raws+1),
                                             size=total_raws-sample_size,replace=False)
                df=pd.read_csv(self.raw_data_path,
                               names=['user_id','item_id','category_id','behavior_type','timestamp'],
                               skiprows=skip_raws)
        except MemoryError:
            print("内存不足，使用分块抽样...")
            df = self._load_with_chunking(sample_size)

        print(f"数据加载完成:{len(df):,}条记录")
        print(f"加载时间:{time.time()-start_time:.2f}秒")
        return df
    def _load_with_chunking(self,sample_size,chunk_size=100000):
        """
        分块加载，内存不足时使用
        :param sample_size:
        :param chunk_size:
        :return:
        """
        chunks=[]
        for chunk in pd.read_csv(self.raw_data_path,
                                 names=['user_id','item_id','category_id','behavior_type','timestamp'],
                                 chunksize=chunk_size):
            chunks.append(chunk)
            if len(chunks)*chunk_size>=sample_size:
                break
        df=pd.concat(chunks,ignore_index=True)
        if len(df) > sample_size:
            df=df.sample(n=sample_size,random_state=42)

        return df
    def validate_data_structure(self,df):
        """
        验证数据结构

        :param df:
        :return:
        """
        print("\n数据结构验证")
        print(f"数据形状:{df.shape}")
        print(f"列名{list(df.columns)}")

        #检查必要列是否存在
        required_columns = ['user_id','item_id','category_id','behavior_type','timestamp']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"缺少必要列: {missing_columns}")

        print("数据验证通过")
        return  True
data_loader = DataLoader()






