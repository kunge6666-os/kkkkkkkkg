from matplotlib import pyplot as plt
import pandas as pd
from data_cleaner import data_cleaner
from data_loader import data_loader
from src.config import FIGURES_PATH


class RetentionAnalyzer:
    def __init__(self):
        self.retention_days=[1,3,7,14,30]#留存天数指标

    def calculate_user_retention(self, df):
        """
        计算用户留存率
        """
        print("=== 用户留存分析 ===")

        # 获取每个用户的首次活跃日期
        user_first_active = df.groupby('user_id')['date'].min().reset_index()
        user_first_active.columns = ['user_id', 'first_active_date']

        # 获取每日活跃用户
        daily_active_users = df.groupby(['date', 'user_id']).size().reset_index()
        daily_active_users = daily_active_users[['date', 'user_id']]

        # 合并首次活跃日期
        user_activity = daily_active_users.merge(user_first_active, on='user_id')

        # 计算每个用户每次活跃距离首次活跃的天数
        user_activity['date'] = pd.to_datetime(user_activity['date'], format='%Y-%m-%d', errors='coerce')
        user_activity['first_active_date'] = pd.to_datetime(user_activity['first_active_date'])
        user_activity['days_since_first'] = (user_activity['date'] - user_activity['first_active_date']).dt.days

        retention_results = {}

        # 计算每日新增用户的后续留存
        cohort_data = self._calculate_cohort_retention(user_activity)
        retention_results['cohort_analysis'] = cohort_data

        # 计算整体留存率
        overall_retention = self._calculate_overall_retention(user_activity)
        retention_results['overall_retention'] = overall_retention

        return retention_results

    def _calculate_cohort_retention(self, user_activity):
        """
        计算同期群留存（Cohort Analysis）
        """
        print("计算同期群留存...")

        # 按周分组（简化版）
        user_activity['cohort_week'] = user_activity['first_active_date'].dt.to_period('W')

        # 创建同期群矩阵
        cohort_data = user_activity.groupby(['cohort_week', 'days_since_first'])['user_id'].nunique().reset_index()
        cohort_pivot = cohort_data.pivot_table(
            index='cohort_week',
            columns='days_since_first',
            values='user_id',
            aggfunc='sum'
        )

        # 计算留存率
        cohort_size = cohort_pivot.iloc[:, 0]  # 第一天的用户数
        retention_matrix = cohort_pivot.divide(cohort_size, axis=0)

        print("同期群留存矩阵（前5行）:")
        print(retention_matrix.head().round(3))

        return retention_matrix

    def _calculate_overall_retention(self, user_activity):
        """
        计算整体留存率
        """
        print("计算整体留存率...")

        # 获取所有新增用户
        new_users = user_activity[user_activity['days_since_first'] == 0]['user_id'].unique()
        total_new_users = len(new_users)

        retention_rates = {}

        for days in self.retention_days:
            # 找到在指定天数后仍然活跃的用户
            retained_users = user_activity[
                (user_activity['days_since_first'] >= days) &
                (user_activity['user_id'].isin(new_users))
                ]['user_id'].nunique()

            retention_rate = retained_users / total_new_users * 100
            retention_rates[f'第{days}天留存率'] = retention_rate

            print(f"  第{days}天留存率: {retention_rate:.2f}% ({retained_users}/{total_new_users})")

        return retention_rates

    def create_retention_visualization(self, retention_results, save=True):
        """
        创建留存分析可视化
        """
        print("生成留存分析图表...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 图表1: 整体留存曲线
        overall_retention = retention_results['overall_retention']
        days = [int(k.replace('第', '').replace('天留存率', '')) for k in overall_retention.keys()]
        rates = list(overall_retention.values())

        ax1.plot(days, rates, marker='o', linewidth=2, markersize=8, color='#FF6B6B')
        ax1.fill_between(days, rates, alpha=0.3, color='#FF6B6B')
        ax1.set_xlabel('天数')
        ax1.set_ylabel('留存率 (%)')
        ax1.set_title('用户留存曲线', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 添加数据标签
        for day, rate in zip(days, rates):
            ax1.text(day, rate + 1, f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

        # 图表2: 留存率柱状图
        colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FF6B6B', '#C44569']
        bars = ax2.bar(range(len(rates)), rates, color=colors, alpha=0.8)
        ax2.set_xlabel('留存周期')
        ax2.set_ylabel('留存率 (%)')
        ax2.set_title('关键周期留存率', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(rates)))
        ax2.set_xticklabels([f'第{day}天' for day in days])

        # 添加数据标签
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        if save:
            save_path = FIGURES_PATH / "user_retention.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 留存图表已保存至: {save_path}")

        plt.show()

        return fig

    # 创建全局实例
retention_analyzer = RetentionAnalyzer()
# df=data_cleaner.create_time_features(data_loader.load_with_sampling())
# retention_analyzer.calculate_user_retention(df)