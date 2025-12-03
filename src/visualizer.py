"""
可视化模块
负责生成专业的数据可视化图标
"""
import matplotlib.pyplot as plt
import seaborn as sns

from src.analyzer import data_analyzer
from src.config import FIGURES_PATH
from src.data_cleaner import data_cleaner
from src.data_loader import data_loader


class DataVisualizer:
    def __init__(self,style='seaborn-v0_8'):
        self.style=style
        self.figures_path=FIGURES_PATH
        self._set_style()

    def _set_style(self):
        """设置绘图风格"""
        plt.style.use(self.style)
        sns.set_palette("husl")

        #设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def create_eda_plots(self,df,analysis_results,save=False):
        """
        创建探索性分析图表
        :param df:
        :param analysis_results:
        :param save:
        :return:
        """
        print("生成探索性分析图表")

        fig=plt.figure(figsize=[20,16])

        #用户行为分布
        plt.subplot(2,3,1)
        self._plot_behavior_distribution(df)

        #24小时活跃分布
        plt.subplot(2,3,2)
        self._plot_hourly_cativity(df)

        #用户活跃度分析
        plt.subplot(2,3,3)
        self._plot_user_activity_distribution(analysis_results.get('user_analysis',{}))

        #时间趋势
        plt.subplot(2,3,4)
        self._polt_daily_trend(df)

        #用户价值分层
        plt.subplot(2,3,5)
        self._polt_user_segments(analysis_results.get('user_analysis',{}))

        #周末vs工作日
        plt.subplot(2,3,6)
        self._plot_weekend_analysis(df)

        plt.tight_layout()

        if save:
            save_path=self.figures_path / "exploratory_analysis.png"
            plt.savefig(save_path,dpi=300,box_inches='tight')
            print(f"图标已保存至:{save_path}")
        plt.show()

    def _plot_behavior_distribution(self,df):
        """绘制用户行为分布"""
        behavior_counts=df['behavior_type'].value_counts()
        colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

        bars=plt.bar(behavior_counts.index,behavior_counts.values,color=colors,alpha=0.8)
        plt.title('用户行为类型分布',fontsize=20,fontweight='bold')
        plt.ylabel('发生次数')

        #添加数据标签
        for bar,count in zip(bars,behavior_counts.values):
            height=bar.get_height()
            percentage=count/len(df)*100
            plt.text(bar.get_x()+bar.get_width()/2.,height+max(behavior_counts.values)*0.01,
                     f'{count:,}\n{percentage:.2f}%',ha='center',va='bottom',fontsize=10)


    def _plot_hourly_cativity(self,df):
        """绘制24小时活跃度分析"""
        hourly_activity=df.groupby('hour').size()

        plt.plot(hourly_activity.index,hourly_activity.values,
                 marker='o',linestyle='-',color='#FF6B6B',markersize=5)
        plt.fill_between(hourly_activity.index,hourly_activity.values,alpha=0.2,color='#FF6B6B')
        plt.fill_between(hourly_activity.index, hourly_activity.values, alpha=0.3, color='#FF6B6B')
        plt.title('24小时用户活跃度分布', fontsize=14, fontweight='bold')
        plt.xlabel('小时')
        plt.ylabel('行为次数')
        plt.grid(True, alpha=0.3)

        # 标记高峰时段
        peak_hour = hourly_activity.idxmax()
        peak_value = hourly_activity.max()
        plt.annotate(f'高峰: {peak_hour}点\n{peak_value:,}次',
                     xy=(peak_hour, peak_value),
                     xytext=(peak_hour + 1, peak_value * 0.8),
                     arrowprops=dict(arrowstyle='->', color='red'),
                     fontweight='bold')

    def _plot_user_activity_distribution(self, user_analysis):
        """绘制用户活跃度分布"""
        if 'user_activity' not in user_analysis:
            plt.text(0.5, 0.5, '无用户活动数据', ha='center', va='center')
            plt.title('用户活跃度分布')
            return

        user_activity = user_analysis['user_activity']

        plt.hist(user_activity['total_actions'], bins=50, color='#4ECDC4',
                 alpha=0.7, edgecolor='black')
        plt.yscale('log')  # 对数尺度处理长尾分布
        plt.title('用户行为频次分布', fontsize=14, fontweight='bold')
        plt.xlabel('用户行为次数')
        plt.ylabel('用户数量（对数）')
        plt.grid(True, alpha=0.3)

    def _polt_daily_trend(self, df):
        """绘制每日趋势"""
        daily_activity = df.groupby('date').size()

        plt.plot(daily_activity.index, daily_activity.values,
                 linewidth=2, color='#45B7D1')
        plt.title('每日活跃趋势', fontsize=14, fontweight='bold')
        plt.xlabel('日期')
        plt.ylabel('日行为次数')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

    def _polt_user_segments(self, user_analysis):
        """绘制用户分层"""
        if 'user_value_segments' not in user_analysis:
            plt.text(0.5, 0.5, '无用户分层数据', ha='center', va='center')
            plt.title('用户价值分层')
            return

        segments = user_analysis['user_value_segments']
        colors = ['#FF6B6B', '#4ECDC4', '#96CEB4']

        plt.pie(segments.values, labels=segments.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        plt.title('用户价值分层', fontsize=14, fontweight='bold')

    def _plot_weekend_analysis(self, df):
        """绘制周末分析"""
        weekend_behavior = df.groupby(['is_weekend', 'behavior_type']).size().unstack(fill_value=0)
        weekend_behavior_percent = weekend_behavior.div(weekend_behavior.sum(axis=1), axis=0)

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        weekend_behavior_percent.plot(kind='bar', ax=plt.gca(), color=colors)
        plt.title('周末vs工作日行为对比', fontsize=14, fontweight='bold')
        plt.xlabel('')
        plt.ylabel('行为比例')
        plt.xticks([0, 1], ['工作日', '周末'], rotation=0)
        plt.legend(title='行为类型', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

    # 创建全局实例
data_visualizer = DataVisualizer()
# processed_df = data_cleaner.creat_time_features(data_loader.load_with_sampling())
# data_visualizer.creat_eda_plots(
#     analysis_results=data_analyzer.basic_analysis(processed_df),
#    df=processed_df
# )

