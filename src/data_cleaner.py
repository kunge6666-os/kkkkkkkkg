from jinja2.utils import missing
import pandas as pd
from pandas.core.algorithms import duplicated

from src.data_loader import data_loader


class DataCleaner:
    def __init__(self):
        self.valid_behaviors=['pv','cart','fav','buy']

    def generate_quailty_report(self,df):
        """
        生成详细的数据质量报告
        :param df:
        :return:
        """
        print("===数据质量检查===")

        report={
            'basic_info':{
                'shape':df.shape,
                'memory_usage_mb':df.memory_usage(deep=True).sum() /1024/1024,
                'columns':list(df.columns)
            },
            'missing_value':df.isnull().sum().to_dict(),
            'data_type':df.dtypes.to_dict(),
            'duplicates':df.duplicated().sum()
        }

        #计算缺失值比例
        missing_percent=(report['missing_value'][col]/len(df)*100
        for col in report['missing_value'])
        report['missing_percent']=dict(zip(report['missing_value'].keys(), missing_percent))

        self._print_quality_report(report)
        return report
    def _print_quality_report(self,report):
        """
        打印质量报告
        :param report:
        :return:
        """
        basic=report["basic_info"]
        print(f"数据形状{basic['shape']}")
        print(f"内存使用{basic['memory_usage_mb']:.2f}MB")
        print(f"重复行使用{report['duplicates']}")

        print("\n缺失值统计:")
        for col,missing in report['missing_value'].items():
            percent=report['missing_percent'][col]
            if missing>0:
                print(f" {col}:{missing}({percent:.2f}%)")
    def optimize_data_types(self,df):
        """
        优化数据类型以减少内存使用
        :param df:
        :return:
        """
        print("\n数据类型使用")

        df_optimized=df.copy()

        #优化整列
        int_columns=['user_id','item_id','category_id']
        for col in int_columns:
            if col in df_optimized.columns:
                df_optimized[col]=df_optimized[col].astype('int32')

        #分类列优化
        if'behavior_type' in df_optimized.columns:
            df_optimized['behavior_type']=df_optimized['behavior_type'].astype('category')
            memory_save=(df.memory_usage(deep=True).sum()-df_optimized.memory_usage(deep=True).sum())/1024/1024
            print(f"内存节省了:{memory_save:.2f}MB")

            return df_optimized

    def create_time_features(self,df):
        """
        创建时间相关特征
        :param df:
        :return:
        """
        print("\n时间特征工程")

        df_features=df.copy()
        print("timestamp统计信息:")
        print(df_features['timestamp'].describe())  # 看最大值、最小值、中位数
        print("\n异常timestamp（小于1000000000或大于31553789759）:")
        abnormal_mask = (df_features['timestamp'] < 1000000000) | (df_features['timestamp'] > 31553789759)
        print(df_features[abnormal_mask]['timestamp'].head(10).tolist())  # 打印异常值

        # 第二步：过滤无效时间戳（只保留2001-2099年的合理数据）
        # 2001-09-09 01:46:40 的timestamp是 1000000000
        # 2099-12-31 23:59:59 的timestamp是 4102444799
        valid_mask = (df_features['timestamp'] >= 1000000000) & (df_features['timestamp'] <= 4102444799)
        df_features = df_features[valid_mask].copy()
        print(f"\n过滤异常timestamp后，剩余数据量: {len(df_features)} 条")
        #转化时间戳
        df_features['datetime']=pd.to_datetime(df_features['timestamp'],unit='s')

        #创建时间特征
        df_features['date']=df_features['datetime'].dt.date
        df_features['hour']=df_features['datetime'].dt.hour
        df_features['day_of_week']=df_features['datetime'].dt.dayofweek
        df_features['is_weekend']=df_features['day_of_week'].isin([5,6]).astype('int8')
        df_features['month']=df_features['datetime'].dt.month
        df_features['day']=df_features['datetime'].dt.day

        #打印时间范围信息
        time_range=df_features['datetime'].max()-df_features['datetime'].min()
        print(f"时间范围:{df_features['datetime'].min()}到:{df_features['datetime'].max()}")
        print(f"覆盖天数:{time_range.days}天")
        print(f"时间特征创建完成: {'datetime','date','hour','day_of_week','month','day','is_weekend','month','day'}")
        return df_features


    def clean_data(self,df):
        """
        执行数据清晰流程
        :param df:
        :return:
        """
        print("\n数据清洗")
        initial_count=len(df)

        #删除重复值
        df_cleaned=df.drop_duplicates()
        duplicates_removed=initial_count-len(df_cleaned)
        print(f"删除重复值:{duplicates_removed}行")

        #验证行为类型
        invalid_mask=~df_cleaned['behavior_type'].isin(self.valid_behaviors)
        initial_count=invalid_mask.sum()
        if initial_count>0:
            print(f"删除无效数据类型:{initial_count}行")
            df_cleaned=df_cleaned[~invalid_mask]

        #3.处理异常值


        final_count=len(df_cleaned)
        retention_rate=final_count/initial_count*100
        print(f"清洗后的数据:{final_count}条")
        print(f"数据保留率:{retention_rate:.2f}%")

        return df_cleaned

data_cleaner=DataCleaner()
#data_cleaner.optimize_data_types(data_loader.load_with_sampling())
#data_cleaner.create_time_features(data_loader.load_with_sampling())




