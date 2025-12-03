"""
用户漏斗分析模块
分析用户从浏览到购买的转化路径
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from numpy import save

from config import FIGURES_PATH
from src.analyzer import data_analyzer
from src.data_loader import data_loader
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class FunnelAnalyzer:
    def __init__(self):
        self.funnel_steps=['pv','cart','fav','buy']
        self.step_names=['浏览','加购','收藏','购买']

    def analyze_conversion_funnel(self,df):
        """
        分析完整的用户转化漏斗
        :param df:
        :return:
        """
        print("===用户漏斗转化分析===")

        funnel_data={}

        #计算每个步骤的独立用户
        for step in self.funnel_steps:
            unique_users=df[df['behavior_type']==step]['user_id'].nunique()
            funnel_data[step]=unique_users

        #计算转化率
        funnel_df=pd.DataFrame({
            '步骤':self.step_names,
            '用户数':[funnel_data[step] for step in self.funnel_steps],
            '转化率':[funnel_data[step]/funnel_data['pv']*100 for step in self.funnel_steps]
        })

        #计算步骤间转化率
        funnel_df['步骤转化率']=0.0
        for i in range(1,len(funnel_df)):
            prev_step=self.funnel_steps[i-1]
            current_step=self.funnel_steps[i]
            conversion=funnel_data[current_step]/funnel_data[prev_step]*100
            funnel_df.loc[i,'步骤转化率']=conversion

        print("转化漏斗数据：")
        print(funnel_df.round(2))
        return funnel_df

    def analyze_funnel_dropoff(self, df, funnel_df):
        """
        分析漏斗流失点
        :param df:
        :param funnel_df:
        :return:
        """
        print("\n===分析漏斗流失点")

        # 校验必要参数，避免后续报错
        if not hasattr(self, 'funnel_steps') or len(self.funnel_steps) < 2:
            raise ValueError("funnel_steps 不能为空且至少包含2个步骤")
        if not hasattr(self, 'step_names') or len(self.step_names) != len(self.funnel_steps):
            raise ValueError("step_names 需与 funnel_steps 长度一致")

        # 计算流失用户
        dropoff_analysis = []
        for i in range(len(funnel_df) - 1):
            current_step = self.funnel_steps[i]
            next_step = self.funnel_steps[i + 1]

            # 完成当前步骤未进行下一步的用户（用户去重，避免重复统计）
            current_user = set(df[df['behavior_type'] == current_step]['user_id'].unique())
            next_user = set(df[df['behavior_type'] == next_step]['user_id'].unique())
            dropped_users = current_user.difference(next_user)

            # 处理当前步骤无用户的边界情况，避免除零错误
            total_current = len(current_user)
            if total_current == 0:
                dropoff_rate = 0.0
                dropped_user_count = 0
            else:
                dropoff_rate = len(dropped_users) / total_current * 100
                dropped_user_count = len(dropped_users)

            step_conversion_rate = 100 - dropoff_rate

            dropoff_analysis.append({
                '从步骤': self.step_names[i],
                '到步骤': self.step_names[i + 1],
                # 修复：存储流失用户数（数字），而非用户ID集合（原命名保留）
                '流失用户数': dropped_user_count,
                '流失率': dropoff_rate,
                '步骤转化率': step_conversion_rate
            })

        dropoff_df = pd.DataFrame(dropoff_analysis)
        print("关键流失点分析:")
        print(dropoff_df.round(2))

        return dropoff_df

    def create_funnel_visualization(self,funnel_df):
        """创建漏斗可视化"""
        print("生成转化漏斗图标")

        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(16,6))

        #图表1：漏斗转化
        y_pos=np.arange(len(funnel_df))
        colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

        bars = ax1.barh(y_pos,funnel_df['用户数'],color=colors,alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(funnel_df['步骤'])
        ax1.set_xlabel('用户数量')
        ax1.set_title('用户转化漏斗',fontsize=14,fontweight='bold')

        #在柱子上添加标签
        for i,(bar,count,rate) in enumerate(zip(bars,funnel_df['用户数'],funnel_df['转化率'])):
            width=bar.get_width()
            ax1.text(width+max(funnel_df['用户数'])*0.01,bar.get_y()+bar.get_height()/2,
                     f'{count:,}\n({rate:.1f}%)',va='center',fontweight='bold')
            #图表2: 步骤转化率
        steps_conversion = funnel_df[1:].copy()  # 从第二步开始
        bars2 = ax2.bar(range(len(steps_conversion)), steps_conversion['步骤转化率'],
                       color=colors[1:], alpha=0.8)
        ax2.set_xlabel('转化步骤')
        ax2.set_ylabel('转化率 (%)')
        ax2.set_title('步骤间转化率', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(steps_conversion)))
        ax2.set_xticklabels([f'{steps_conversion.iloc[i-1]["步骤"]}→{steps_conversion.iloc[i]["步骤"]}'
                           for i in range(len(steps_conversion))], rotation=45)

        # 添加数据标签
        for bar, rate in zip(bars2, steps_conversion['步骤转化率']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        if save:
            save_path = FIGURES_PATH / "conversion_funnel.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 漏斗图表已保存至: {save_path}")

        plt.show()

        return fig

#创建全局实例
funnel_analyzer = FunnelAnalyzer()
# df=data_loader.load_with_sampling()
# funnel_df = funnel_analyzer.analyze_conversion_funnel(df)
#
# #3. 调用第二个方法：分析漏斗流失点（依赖第一个方法的结果）
# dropoff_df = funnel_analyzer.analyze_funnel_dropoff(df,funnel_df)
#
# #4. 调用第三个方法：创建漏斗可视化（依赖第一个方法的结果）
# fig = funnel_analyzer.creat_finnel_visualization(funnel_df)


