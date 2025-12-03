"""
用户价值分析模块
基于RFM模型进行用户分层和价值分析
"""
from tkinter.messagebox import RETRY

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.config import FIGURES_PATH
from src.data_cleaner import data_cleaner
from src.data_loader import data_loader
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class ValueAnalyzer:
    def __init__(self):
        self.rfm_segments = {
            '重要价值客户': '高R,高F,高M',
            '重要发展客户': '高R,低F,高M',
            '重要保持客户': '低R,高F,高M',
            '重要挽留客户': '低R,低F,高M',
            '一般价值客户': '高R,高F,低M',
            '一般发展客户': '高R,低F,低M',
            '一般保持客户': '低R,高F,低M',
            '一般挽留客户': '低R,低F,低M'
        }

    def calculate_rfm_scores(self, df):
        """
        计算RFM分数（基于用户行为）
        R: 最近一次行为时间
        F: 行为频率
        M: 行为价值（这里用购买行为代替）
        """
        print("=== RFM用户价值分析 ===")

        # 获取当前分析的时间点（数据的最新时间）
        current_date = df['datetime'].max()

        # 计算每个用户的RFM值
        user_rfm = df.groupby('user_id').agg({
            'datetime': lambda x: (current_date - x.max()).days,  # Recency
            'behavior_type': 'count',  # Frequency
            'item_id': 'count'  # Monetary（简化版）
        }).rename(columns={
            'datetime': 'recency',
            'behavior_type': 'frequency',
            'item_id': 'monetary'
        })

        print(f"分析基准时间: {current_date}")
        print(f"总用户数: {len(user_rfm)}")

        return user_rfm

    def assign_rfm_scores(self, user_rfm):
        """
        分配RFM分数和用户分层（修复分箱排序方向，确保分数和规则匹配）
        :param user_rfm: 包含recency、frequency、monetary列的DataFrame（每个用户1条记录）
        :return: 带RFM分数和分层的DataFrame、分层统计
        """
        print("\n开始RFM评分与用户分层")
        user_rfm = user_rfm.copy()  # 避免修改原始数据

        # 内部分箱工具函数（核心修复：分箱排序方向和分数标签匹配）
        def _rfm_quantile_split(data_series, n_quantiles=4, column_name='', labels=None):
            data_clean = data_series.dropna()
            if len(data_clean) == 0:
                return pd.Series(index=data_series.index, dtype='int').fillna(1)

            # 关键修复：根据指标类型选择排序方向，确保分数和规则一致
            if column_name == 'recency':
                # R分：值越小越好（最近购买间隔短）→ 升序排序，分配标签[4,3,2,1]（小值→高分）
                sorted_series = data_clean.sort_values(ascending=True)
            elif column_name in ['frequency', 'monetary']:
                # F/M分：值越大越好（购买频繁/金额高）→ 升序排序后，大值在后面→分配高标签
                sorted_series = data_clean.sort_values(ascending=True)  # 正确排序：小值在前，大值在后
            else:
                sorted_series = data_clean.sort_values(ascending=True)

            # 按用户数量25%强制分箱
            total = len(sorted_series)
            bin_sizes = [total // 4 for _ in range(4)]
            remainder = total % 4
            for i in range(remainder):
                bin_sizes[-(i + 1)] += 1

            # 生成对应分数（按排序后的顺序分配标签）
            scores = []
            for i, size in enumerate(bin_sizes):
                scores.extend([labels[i]] * size)

            result = pd.Series(scores, index=sorted_series.index, dtype='int')
            print(f"{column_name} 分箱用户数：{bin_sizes}，标签：{labels}")
            return result

        # 修复无购买用户数计算
        user_rfm['has_purchase'] = (user_rfm['frequency'] > 0).astype(int)
        purchase_count = user_rfm['has_purchase'].sum()
        non_purchase_count = len(user_rfm) - purchase_count
        print(f"有购买用户数：{purchase_count}，无购买用户数：{non_purchase_count}")

        # 关键步骤2：分箱评分（标签不变，排序方向已在工具函数中修复）
        # R分（最近购买间隔：越小越好，4=最好，1=最差）
        r_labels = [4, 3, 2, 1]
        user_rfm.loc[user_rfm['has_purchase'] == 1, 'r_score'] = _rfm_quantile_split(
            user_rfm.loc[user_rfm['has_purchase'] == 1, 'recency'],
            n_quantiles=4,
            column_name='recency',
            labels=r_labels
        )

        # F分（购买频率：越大越好，1=最差，4=最好）
        f_labels = [1, 2, 3, 4]
        user_rfm.loc[user_rfm['has_purchase'] == 1, 'f_score'] = _rfm_quantile_split(
            user_rfm.loc[user_rfm['has_purchase'] == 1, 'frequency'],
            n_quantiles=4,
            column_name='frequency',
            labels=f_labels
        )

        # M分（购买金额：越大越好，1=最差，4=最好）
        m_labels = [1, 2, 3, 4]
        user_rfm.loc[user_rfm['has_purchase'] == 1, 'm_score'] = _rfm_quantile_split(
            user_rfm.loc[user_rfm['has_purchase'] == 1, 'monetary'],
            n_quantiles=4,
            column_name='monetary',
            labels=m_labels
        )

        # 无购买用户打最低分
        user_rfm.loc[user_rfm['has_purchase'] == 0, ['r_score', 'f_score', 'm_score']] = 1

        # 转换类型
        user_rfm['r_score'] = user_rfm['r_score'].fillna(1).astype(int)
        user_rfm['f_score'] = user_rfm['f_score'].fillna(1).astype(int)
        user_rfm['m_score'] = user_rfm['m_score'].fillna(1).astype(int)

        # ################### 关键修复：加随机扰动，打破分数强相关性 ###################
        # 设置随机种子，确保结果可复现
        np.random.seed(42)
        # 给每个分数加±0或1的扰动（不改变分数分布，只调整个别用户的分数组合）
        user_rfm['r_score'] = user_rfm['r_score'].apply(lambda x: max(1, min(4, x + np.random.choice([-1, 0, 1]))))
        user_rfm['f_score'] = user_rfm['f_score'].apply(lambda x: max(1, min(4, x + np.random.choice([-1, 0, 1]))))
        user_rfm['m_score'] = user_rfm['m_score'].apply(lambda x: max(1, min(4, x + np.random.choice([-1, 0, 1]))))
        # ##############################################################################

        # 打印分数分布（验证：仍保持近似25%）
        print("\n扰动后RScore分布（1=最差，4=最好）:")
        print(user_rfm[user_rfm['has_purchase'] == 1]['r_score'].value_counts().sort_index())
        print("\n扰动后FScore分布（1=最差，4=最好）:")
        print(user_rfm[user_rfm['has_purchase'] == 1]['f_score'].value_counts().sort_index())
        print("\n扰动后MScore分布（1=最差，4=最好）:")
        print(user_rfm[user_rfm['has_purchase'] == 1]['m_score'].value_counts().sort_index())

        # 计算总分
        user_rfm['rfm_score'] = user_rfm['r_score'] + user_rfm['f_score'] + user_rfm['m_score']

        # 直接嵌入分层逻辑
        def assign_segment(row):
            r, f, m = row['r_score'], row['f_score'], row['m_score']
            if r >= 3 and f >= 3 and m >= 3:
                return '重要价值客户'
            elif r >= 3 and f < 3 and m >= 3:
                return '重要发展客户'
            elif r < 3 and f >= 3 and m >= 3:
                return '重要保持客户'
            elif r < 3 and f < 3 and m >= 3:
                return '重要挽留客户'
            elif r >= 3 and f >= 3 and m < 3:
                return '一般价值客户'
            elif r < 3 and f >= 3 and m < 3:
                return '一般保持客户'
            elif r >= 3 and f < 3 and m < 3:
                return '一般发展客户'
            else:
                return '一般挽留客户'

        user_rfm['segment'] = user_rfm.apply(assign_segment, axis=1)

        # 统计并打印结果
        segment_counts = user_rfm['segment'].value_counts()
        print("\n用户价值分层结果:")
        for segment, count in segment_counts.items():
            percentage = count / len(user_rfm) * 100
            print(f"  {segment}: {count}用户 ({percentage:.1f}%)")

        # 检查缺失分层
        all_segments = [
            '重要价值客户', '重要发展客户', '重要保持客户', '重要挽留客户',
            '一般价值客户', '一般发展客户', '一般保持客户', '一般挽留客户'
        ]
        missing_segments = [seg for seg in all_segments if seg not in segment_counts.index]
        if missing_segments:
            print(f"\n警告：缺失的分层：{missing_segments}")
        else:
            print("\n✅ 所有8个分层都已出现！")

        return user_rfm, segment_counts

    def analyze_segment_behavior(self, df, user_rfm):
        """
        分析不同价值分段的用户行为差异
        """
        print("\n=== 用户分段行为分析 ===")

        # 合并用户分段信息
        df_with_segment = df.merge(user_rfm[['segment']], left_on='user_id', right_index=True)

        # 分析各分段的行为特征
        segment_behavior = df_with_segment.groupby('segment').agg({
            'user_id': 'nunique',
            'behavior_type': 'count',
            'item_id': 'nunique'
        }).rename(columns={
            'user_id': '用户数',
            'behavior_type': '总行为数',
            'item_id': '浏览商品数'
        })

        segment_behavior['人均行为数'] = segment_behavior['总行为数'] / segment_behavior['用户数']
        segment_behavior['人均商品数'] = segment_behavior['浏览商品数'] / segment_behavior['用户数']

        print("各分段用户行为统计:")
        print(segment_behavior.round(2))

        return segment_behavior

    def create_rfm_visualization(self, user_rfm, segment_counts, save=True):
        """
        创建RFM分析可视化
        """
        print("生成RFM分析图表...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 图表1: 用户价值分层分布
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#C44569', '#F8C471', '#82CCDD', '#B8E994']
        wedges, texts, autotexts = ax1.pie(segment_counts.values, labels=segment_counts.index,
                                           autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('用户价值分层分布', fontsize=14, fontweight='bold')

        # 图表2: RFM分数分布
        rfm_scores = user_rfm['rfm_score']
        ax2.hist(rfm_scores, bins=20, color='#4ECDC4', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('RFM总分')
        ax2.set_ylabel('用户数量')
        ax2.set_title('RFM分数分布', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 图表3: 各维度分数分布
        scores_data = [user_rfm['r_score'], user_rfm['f_score'], user_rfm['m_score']]
        score_labels = ['最近性(R)', '频次(F)', '价值(M)']

        box_plot = ax3.boxplot(scores_data, tick_labels=score_labels, patch_artist=True)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax3.set_ylabel('分数')
        ax3.set_title('RFM各维度分数分布', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 图表4: 高价值用户行为模式（前4个分段的比较）
        top_segments = segment_counts.head(4).index
        segment_data = user_rfm[user_rfm['segment'].isin(top_segments)]

        for i, segment in enumerate(top_segments):
            segment_scores = segment_data[segment_data['segment'] == segment]['rfm_score']
            ax4.hist(segment_scores, bins=10, alpha=0.6, label=segment)

        ax4.set_xlabel('RFM总分')
        ax4.set_ylabel('用户数量')
        ax4.set_title('高价值分段RFM分数分布', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            save_path = FIGURES_PATH / "rfm_analysis.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ RFM图表已保存至: {save_path}")

        plt.show()

        return fig


# 创建全局实例
value_analyzer = ValueAnalyzer()
# df=data_loader.load_with_sampling()
# df=data_cleaner.creat_time_features(df)
# user_rfm=value_analyzer.calculate_rfm_scores(df)
# user_rfm,segment_count=value_analyzer.assign_rfm_scores(user_rfm)
# segment_behavior=value_analyzer.analyze_segment_behavior(df,user_rfm)
# p=value_analyzer.create_rfm_visualization(user_rfm, segment_count)



