"""时间序列分析，分析趋势，周期性和季节性模式"""
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from src.config import FIGURES_PATH
from src.data_cleaner import data_cleaner
from src.data_loader import data_loader

# 在绘制图表前，添加以下代码（二选一即可，根据系统调整）
# Windows 系统常用：微软雅黑
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# macOS 系统常用：苹方
# plt.rcParams['font.sans-serif'] = ['PingFang SC']
# Linux 系统常用：文泉驿微米黑
# plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']

# 解决负号显示异常（可选，若图表有负数值）
plt.rcParams['axes.unicode_minus'] = False



class TimeSeriesAnalyzer:
    def __init__(self):
        self.decomposition=None

    def prepare_time_series_data(self,df,freq='D'):
        """准备时间序列数据"""
        print("===时间序列数据准备===")

        #创建日级时间序列
        daily_metrics=df.groupby('date').agg({
            'user_id':'nunique',
            'item_id':'nunique',
            'behavior_type':'count'
        }).rename(columns={
            'user_id':'dau',
            'item_id':'daily_items',
            'behavior_type':'daily_actions'
        })
        #购买行为时间序列
        daily_purchase=df[df['behavior_type']=='buy'].groupby('date').agg({
            'user_id':'nunique',
            'behavior_type':'count'
        }).rename(columns={
            'user_id':'daily_buyers',
            'behavior_type':'daily_purchases'
        })

        #合并数据
        time_series_data=daily_metrics.merge(daily_purchase, on='date', how='left').fillna(0)

        #计算衍生指标
        time_series_data['purchase_conversion']=time_series_data['daily_buyers']/time_series_data['dau']*100
        time_series_data['avg_actions_per_user']=time_series_data['daily_actions']/time_series_data['dau']*100

        print(f"时间序列长度{len(time_series_data)}天")
        print(f"时间范围{time_series_data.index.min()}到{time_series_data.index.max()}")

        return time_series_data
    def analyze_trends(self,time_series_data):
        """分析时间序列趋势"""
        print("\n时间序列趋势")
        trend_analysis={}
        #print("数据预览：")
        #print(time_series_data[['dau', 'daily_actions', 'daily_purchases']].head(10))
        #print("\n数据统计：")
        #print(time_series_data[['dau', 'daily_actions', 'daily_purchases']].describe())
        #print("数据的日期索引：", time_series_data.index.tolist())
        #计算关键指标的增长率
        metrics=['dau','daily_actions','daily_purchases']
        for metric in metrics:

            metric_series = time_series_data[metric]


            metric_series_clean = metric_series.dropna()

            # 确保过滤后有至少2个数据点（避免只有1个值无法计算增长）
            if len(metric_series_clean) < 2:
                print(f"指标 {metric} 有效数据不足2个，无法计算增长率")
                continue

            first_val = metric_series_clean.iloc[0]
            last_val = metric_series_clean.iloc[-1]

            # 处理首值为0的情况
            if first_val == 0:
                if last_val > 0:
                    # 场景1：从0增长到正数 → 标注“从0启动增长”，或用“绝对增长量”替代增长率
                    growth_result = f"从0增长至{last_val}（绝对增长量：{last_val}）"
                elif last_val == 0:
                    # 场景2：首末值均为0 → 无增长（增长率0%）
                    growth_result = "0%（首末值均为0，无增长）"
                else:
                    # 场景3：首值0、尾值负数（你的指标不可能出现，DAU/行为数/购买数均非负）
                    growth_result = "无效数据（尾值为负）"
            else:
                # 常规情况：首值非0 → 计算正常增长率（保留2位小数）
                growth_rate = (last_val - first_val) / first_val * 100
                growth_result = f"{growth_rate:.2f}%"
            trend_analysis[f'{metric}_增长率']=growth_result

            print(f"指标 {metric} 增长率：{growth_result}")

        #计算移动平均平滑趋势
        for metric in metrics:
            time_series_data[f'{metric}_ma7']=time_series_data[metric].rolling(window=7).mean()
            time_series_data[f'{metric}_ma30']=time_series_data[metric].rolling(window=30).mean()

        return trend_analysis,time_series_data

    def analyze_seasonality(self, time_series_data, metric='daily_actions'):
        """
        分析季节性和周期性模式
        """
        print(f"\n=== {metric} 季节性分析 ===")

        # 确保数据是等间隔的
        ts_data = time_series_data[metric].dropna()

        if len(ts_data) < 30:
            print("数据量不足进行季节性分解")
            return None

        try:
            # 季节性分解
            decomposition = seasonal_decompose(ts_data, model='additive', period=7)
            self.decomposition = decomposition

            # 计算季节性强度
            seasonal_strength = decomposition.seasonal.std() / ts_data.std()
            print(f"季节性强度: {seasonal_strength:.3f}")

            return decomposition
        except Exception as e:
            print(f"季节性分解失败: {e}")
            return None

    def stationarity_test(self,time_series_data,metric='daily_actions'):
        """检验时间序列平稳性"""
        print(f"\n{metric}平稳性检验")
        ts_data=time_series_data[metric].dropna()

        #ADF检验
        result=adfuller(ts_data)
        print(f"ADF统计量{result[0]:.4f}")
        print(f"p值{result[1]:.4f}")
        print(f"临界值：")

        for key,value in result[4].items():
            print(f"{key}:{value:.4f}")

        is_stationary=result[1]<=0.05
        print(f"序列是否平稳{'是'if is_stationary else'否'}")

        return {
            'adf_statistic':result[0],
            'p_value':result[1],
            'is_stationary':is_stationary

        }
    def analyze_weekly_patterns(self,time_series_data):
        """分析周内模式"""
        print("周内模式分析")
        #添加星期几信息
        ts_with_weekday=time_series_data.copy()
        ts_with_weekday.index=pd.to_datetime(ts_with_weekday.index)
        ts_with_weekday['weekday']=ts_with_weekday.index.dayofweek
        ts_with_weekday['is_weekday']=ts_with_weekday['weekday'].isin([5,6])

        #按星期几聚合
        weekday_patterns=ts_with_weekday.groupby('weekday').agg({
            'dau':'mean',
            'daily_actions':'mean',
            'daily_purchases':'mean',
            'purchase_conversion':'mean',
        })
        weekday_names=['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        weekday_patterns.index=[weekday_names[i] for i in weekday_patterns.index]

        print("周内模式（平均值）:")
        print(weekday_patterns.round(2))
        return weekday_patterns

    def create_time_series_visualization(self,time_series_data,decomposition=None,save=True):
        """创建时间序列可视化"""
        print("生成时间序列分析图表")
        if decomposition is not None:
            fig,axes=plt.subplots(4,2,figsize=(20,16))
        else:
            fig,axes=plt.subplots(2,2,figsize=(20,12))
            axes=axes.flatten()

        #图表1:DAU趋势
        axes[0,0].plot(time_series_data.index,time_series_data['dau'],
                       label='日活用户',color='#FF6B6B',linewidth=2)
        axes[0, 0].plot(time_series_data.index, time_series_data['dau_ma7'],
                        label='7日移动平均', color='red', linewidth=2, linestyle='--')
        axes[0, 0].set_title('日活跃用户趋势', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('用户数')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 图表2: 行为量趋势
        axes[0, 1].plot(time_series_data.index, time_series_data['daily_actions'],
                        label='日行为量', color='#4ECDC4', linewidth=2)
        axes[0, 1].plot(time_series_data.index, time_series_data['daily_actions_ma7'],
                        label='7日移动平均', color='teal', linewidth=2, linestyle='--')
        axes[0, 1].set_title('用户行为量趋势', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('行为次数')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 图表3: 购买趋势
        axes[1, 0].plot(time_series_data.index, time_series_data['daily_purchases'],
                        label='日购买量', color='#45B7D1', linewidth=2)
        axes[1, 0].plot(time_series_data.index, time_series_data['daily_purchases_ma7'],
                        label='7日移动平均', color='blue', linewidth=2, linestyle='--')
        axes[1, 0].set_title('购买行为趋势', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('购买次数')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 图表4: 转化率趋势
        axes[1, 1].plot(time_series_data.index, time_series_data['purchase_conversion'],
                        label='购买转化率', color='#96CEB4', linewidth=2)
        axes[1, 1].set_title('购买转化率趋势', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('转化率 (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 如果有季节性分解结果，显示分解图表
        if decomposition is not None:
            # 原始序列
            axes[2, 0].plot(decomposition.observed)
            axes[2, 0].set_title('原始序列', fontsize=12)
            axes[2, 0].grid(True, alpha=0.3)

            # 趋势成分
            axes[2, 1].plot(decomposition.trend)
            axes[2, 1].set_title('趋势成分', fontsize=12)
            axes[2, 1].grid(True, alpha=0.3)

            # 季节性成分
            axes[3, 0].plot(decomposition.seasonal)
            axes[3, 0].set_title('季节性成分', fontsize=12)
            axes[3, 0].grid(True, alpha=0.3)

            # 残差
            axes[3, 1].plot(decomposition.resid)
            axes[3, 1].set_title('残差成分', fontsize=12)
            axes[3, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            save_path = FIGURES_PATH / "time_series_analysis.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 时间序列图表已保存至: {save_path}")

        plt.show()

        return fig

# 创建全局实例
time_series_analyzer = TimeSeriesAnalyzer()
# df=data_loader.load_with_sampling()
#date=data_cleaner.creat_time_features(df)
#time_series_data = time_series_analyzer.prepare_time_series_date(date)
#
#  #2.2 趋势分析（计算增长率+移动平均）
# trend_analysis, up_time_series_data = time_series_analyzer.analyze_trends(time_series_data)
#
#  #2.3 季节性分析（以日行为量为例，可修改metric参数）
# decomposition = time_series_analyzer.analyze_seasonality(up_time_series_data, metric='daily_actions')
#
#  #2.4 平稳性检验（可选，验证数据平稳性）
# stationarity_result = time_series_analyzer.stationarity_test(up_time_series_data, metric='daily_actions')
#
#  #2.5 周内模式分析（可选，分析星期分布规律）
# weekday_patterns = time_series_analyzer.analyze_weekly_patterns(time_series_data)
#
#  #------------------------------
#  #步骤3：生成并显示四个核心图表+季节性分解图表
#  #------------------------------
# time_series_analyzer.create_time_series_visualization(
#     time_series_data=time_series_data,
#     decomposition=decomposition,
#     save=True  # True：保存图表到配置的FIGURES_PATH，False：仅显示不保存
# )









