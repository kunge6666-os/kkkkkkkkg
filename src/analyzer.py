import pandas as pd
import numpy as np

from src.data_cleaner import data_cleaner
from src.data_loader import data_loader

"""
分析模块
负责数据分析和统计计算
"""
class DataAnalyzer(object):
    def __init__(self):
        self.behavior_types=['pv','cart','fav','buy']

    def basic_analysis(self,df):
        """
        基础统计分析
        :param df:
        :return:
        """
        print("基础统计分析")
        analysis_results={}
        #用户行为分布
        behavior_counts=df['behavior_type'].value_counts()
        analysis_results['behavior_distribution']=behavior_counts

        total_actions=len(df)
        print("用户行为分析")
        for behavior,count in behavior_counts.items():
            percentage=count/total_actions*100
            print(f"{behavior}:{count:,}次{percentage:.2f}%")
            analysis_results[f'{behavior}_count']=count
            analysis_results[f'{behavior}_percent']=percentage

        #基础指标
        analysis_results['total_users'] = df['user_id'].nunique()
        analysis_results['total_items'] = df['item_id'].nunique()
        analysis_results['total_categories'] = df['category_id'].nunique()
        analysis_results['total_actions'] = total_actions

        print(f"\n基础指标")
        print(f"总用户数{analysis_results['total_users']:,}")
        print(f"总商品数{analysis_results['total_items']:,}")
        print(f"总类目数{analysis_results['total_categories']:,}")
        print(f"总行为数{analysis_results['total_actions']:,}")

        return analysis_results

    def user_behavior_analysis(self,df):
        """
        用户行为深度分析
        :param df:
        :return:
        """
        user_analysis={}

        #用户活跃度分析
        user_activity=df.groupby('user_id').agg({
            'behavior_type':'count',
            'date':'nunique',
            'item_id':'nunique',
            'category_id':'nunique',
        }).rename(columns={
            'behavior_type':'total_actions',
            'date':'active_days',
            'item_id':'unique_items',
            'category_id':'unique_categories',
        })
        user_analysis['user_activity']=user_activity

        print("用户活跃度统计")
        print(f"  平均每个用户行为数: {user_activity['total_actions'].mean():.1f}")
        print(f"  平均活跃天数: {user_activity['active_days'].mean():.1f}")
        print(f"  平均浏览商品数: {user_activity['unique_items'].mean():.1f}")
        print(f"  最活跃用户行为数: {user_activity['total_actions'].max():,}")

        #用户价值分层（简单版）
        user_analysis['user_value_segments']=self._segment_user(user_activity)
        return user_analysis

    def _segment_user(self,user_activity):
        """
        简单用户分层
        :param user_activity:
        :return:
        """
        #基于行为数和活跃天数分层
        conditions = [
            (user_activity['total_actions']>=user_activity['total_actions'].quantile(0.8))&
            (user_activity['active_days']<=user_activity['active_days'].quantile(0.8)),
            (user_activity['total_actions']>=user_activity['total_actions'].quantile(0.5)),
            (user_activity['total_actions']<user_activity['total_actions'].quantile(0.5))

        ]
        segments=['高价值用户','中等价值用户','低价值用户']

        user_activity['segment']=np.select(conditions,segments,default='低价值用户')

        segments_count=user_activity['segment'].value_counts()
        print("\n用户价值分层:")
        for segment,count in segments_count.items():
            percentage=count/len(user_activity)*100
            print(f"{segment}:{count} ({percentage:.2f}%)")

            return segments_count

    def time_based_analysis(self,df):
            """
            时间维度分析
            :param self:
            :param df:
            :return:
            """
            print("时间维度分析")

            time_analysis={}
            #24小时活跃度
            hourly_activity = df.groupby('hour').size()
            time_analysis['hourly_activity']=hourly_activity

            #周活跃度分析
            daily_activity = df.groupby('date').size()
            time_analysis['daily_activity'] = daily_activity

            # 周末vs工作日
            weekend_activity = df.groupby('is_weekend').size()
            time_analysis['weekend_activity'] = weekend_activity

            print("时间模型发现:")
            peak_hour=hourly_activity.idxmax()
            print(f"  高峰时段: {peak_hour}点 ({hourly_activity.max():,}次行为)")
            print(f"  周末活跃比例: {weekend_activity.get(1, 0) / len(df) * 100:.1f}%")

            return time_analysis
data_analyzer=DataAnalyzer()
#data_analyzer.time_based_analysis(data_cleaner.creat_time_features(data_loader.load_with_sampling()))

