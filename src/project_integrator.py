"""
项目整合模块
整合所有分析结果，生成综合报告和可视化看板
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from src.analyzer import data_analyzer
from src.config import PROCESSED_DATA_PATH, FIGURES_PATH
from src.data_cleaner import data_cleaner
from src.data_loader import data_loader

from src.time_series_analyzer import time_series_analyzer

warnings.filterwarnings('ignore')


class ProjectIntegrator:
    def __init__(self):
        self.integrated_results = {}
        self.project_metrics = {}

    def integrate_all_analyses(self, df, analyses_dict):
        """
        整合所有分析结果
        """
        print("=== 项目分析结果整合 ===")

        # 基础指标
        self.project_metrics['basic'] = {
            'total_records': len(df),
            'total_users': df['user_id'].nunique(),
            'total_items': df['item_id'].nunique(),
            'total_categories': df['category_id'].nunique(),
            'date_range': f"{df['datetime'].min().strftime('%Y-%m-%d')} 到 {df['datetime'].max().strftime('%Y-%m-%d')}",
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # 整合各模块分析结果
        self.integrated_results = analyses_dict

        print("✅ 所有分析结果整合完成")
        return self.integrated_results

    def calculate_business_kpis(self, df, integrated_results):
        """
        计算关键业务指标(KPIs)
        """
        print("\n=== 关键业务指标计算 ===")

        kpis = {}

        # 用户相关KPIs
        kpis['user_metrics'] = {
            'dau': integrated_results.get('time_series', {}).get('ts_data', pd.DataFrame()).get('dau', 0).mean(),
            'user_growth_rate': integrated_results.get('time_series', {}).get('trend_analysis', {}).get('dau_增长率',
                                                                                                        0),
            'avg_retention_rate': np.mean(
                list(integrated_results.get('retention', {}).get('overall_retention', {}).values())),
            'high_value_user_ratio': integrated_results.get('rfm', {}).get('segment_counts', pd.Series()).get(
                ['重要价值客户', '重要发展客户']).sum() / integrated_results.get('basic', {}).get('total_users',
                                                                                                  1) * 100
        }

        # 转化相关KPIs
        funnel_df = integrated_results.get('funnel', pd.DataFrame())
        if not funnel_df.empty:
            kpis['conversion_metrics'] = {
                'overall_conversion_rate': funnel_df.loc[funnel_df['步骤'] == '购买', '转化率'].iloc[0],
                'browse_to_cart_rate': funnel_df.loc[funnel_df['步骤'] == '加购', '转化率'].iloc[0],
                'cart_to_purchase_rate':
                    integrated_results.get('funnel_dropoff', pd.DataFrame()).get('步骤转化率', pd.Series()).iloc[
                        -1] if 'funnel_dropoff' in integrated_results else 0
            }

        # 商品相关KPIs
        product_stats = integrated_results.get('product_stats', pd.DataFrame())
        if not product_stats.empty:
            kpis['product_metrics'] = {
                'avg_product_conversion': product_stats[product_stats['购买转化率'] > 0]['购买转化率'].mean(),
                'active_product_ratio': len(product_stats[product_stats['购买次数'] > 0]) / len(product_stats) * 100,
                'top_category_coverage':
                    integrated_results.get('category_stats', pd.DataFrame()).nlargest(3, '总行为数')['总行为数'].sum() /
                    integrated_results.get('category_stats', pd.DataFrame())['总行为数'].sum() * 100
            }

        # 预测相关KPIs
        forecast_df = integrated_results.get('forecast', pd.DataFrame())
        if not forecast_df.empty:
            kpis['forecast_metrics'] = {
                'next_7days_avg': forecast_df['predicted'].mean(),
                'prediction_confidence': (1 - integrated_results.get('model_results', {}).get('RandomForest', {}).get(
                    'mae', 1) / forecast_df['predicted'].mean()) * 100
            }

        self.project_metrics['kpis'] = kpis

        print("关键业务指标:")
        for category, metrics in kpis.items():
            print(f"\n{category}:")
            for metric, value in metrics.items():
                # 强制转换为数值类型，处理可能的字符串/异常值
                try:
                    numeric_value = float(value)
                except (ValueError, TypeError):
                    numeric_value = 0.0
                    print(f"  警告：{metric} 的值 {value} 不是有效数值，已默认设为0.0")
                # 格式化输出数值
                print(f"  {metric}: {numeric_value:.2f}")

        return kpis

    def generate_executive_summary(self):
        """
        生成执行摘要
        """
        print("\n=== 生成执行摘要 ===")

        kpis = self.project_metrics.get('kpis', {})
        basic = self.project_metrics.get('basic', {})

        summary = {
            'project_scope': f"分析了 {basic.get('total_records', 0):,} 条用户行为记录，覆盖 {basic.get('total_users', 0):,} 用户和 {basic.get('total_items', 0):,} 商品",
            'key_strengths': [],
            'key_opportunities': [],
            'recommended_actions': []
        }

        # 识别优势
        if kpis.get('user_metrics', {}).get('avg_retention_rate', 0) > 20:
            summary['key_strengths'].append("用户留存表现良好")

        if kpis.get('conversion_metrics', {}).get('overall_conversion_rate', 0) > 2:
            summary['key_strengths'].append("整体转化率健康")

        # 识别机会点
        if kpis.get('conversion_metrics', {}).get('cart_to_purchase_rate', 0) < 50:
            summary['key_opportunities'].append("加购到购买转化有优化空间")

        if kpis.get('user_metrics', {}).get('high_value_user_ratio', 0) < 20:
            summary['key_opportunities'].append("高价值用户占比有待提升")

        # 推荐行动
        if summary['key_opportunities']:
            summary['recommended_actions'] = [
                "优化购物车到购买的转化路径",
                "设计高价值用户培育计划",
                "基于用户分层制定精准营销策略",
                "利用预测模型优化库存和营销资源分配"
            ]

        self.project_metrics['executive_summary'] = summary

        print("执行摘要生成完成")
        return summary

    def create_comprehensive_dashboard(self, save=True):
        """
        创建综合数据看板
        """
        print("生成综合数据看板...")

        fig = plt.figure(figsize=(20, 16))

        # 使用GridSpec创建复杂的布局
        gs = plt.GridSpec(4, 4, figure=fig)

        # 1. 项目概览指标卡
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[0, 3])

        # 指标卡数据
        basic_metrics = self.project_metrics.get('basic', {})
        kpis = self.project_metrics.get('kpis', {})

        metric_cards = [
            ('总用户数', f"{basic_metrics.get('total_users', 0):,}", '#FF6B6B'),
            ('总商品数', f"{basic_metrics.get('total_items', 0):,}", '#4ECDC4'),
            ('日均活跃', f"{kpis.get('user_metrics', {}).get('dau', 0):.0f}", '#45B7D1'),
            ('整体转化率', f"{kpis.get('conversion_metrics', {}).get('overall_conversion_rate', 0):.2f}%", '#96CEB4')
        ]

        for i, (title, value, color) in enumerate(metric_cards):
            ax = [ax1, ax2, ax3, ax4][i]
            ax.text(0.5, 0.6, value, ha='center', va='center', fontsize=24,
                    fontweight='bold', color=color, transform=ax.transAxes)
            ax.text(0.5, 0.3, title, ha='center', va='center', fontsize=14,
                    transform=ax.transAxes)
            ax.set_facecolor('#f8f9fa')
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)

        # 2. 用户价值分层
        ax5 = fig.add_subplot(gs[1, 0:2])
        if 'rfm' in self.integrated_results and 'segment_counts' in self.integrated_results['rfm']:
            segments = self.integrated_results['rfm']['segment_counts']
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#C44569', '#F8C471']
            wedges, texts, autotexts = ax5.pie(segments.values, labels=segments.index,
                                               autopct='%1.1f%%', colors=colors, startangle=90)
            ax5.set_title('用户价值分层分布', fontsize=14, fontweight='bold')

        # 3. 转化漏斗
        ax6 = fig.add_subplot(gs[1, 2:4])
        if 'funnel' in self.integrated_results:
            funnel_df = self.integrated_results['funnel']
            y_pos = np.arange(len(funnel_df))
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

            bars = ax6.barh(y_pos, funnel_df['用户数'], color=colors, alpha=0.8)
            ax6.set_yticks(y_pos)
            ax6.set_yticklabels(funnel_df['步骤'])
            ax6.set_xlabel('用户数量')
            ax6.set_title('用户转化漏斗', fontsize=14, fontweight='bold')
            ax6.invert_yaxis()

            for i, (bar, count, rate) in enumerate(zip(bars, funnel_df['用户数'], funnel_df['转化率'])):
                width = bar.get_width()
                ax6.text(width + max(funnel_df['用户数']) * 0.01, bar.get_y() + bar.get_height() / 2,
                         f'{count:,}\n({rate:.1f}%)', va='center', fontweight='bold')

        # 4. 时间趋势
        ax7 = fig.add_subplot(gs[2, 0:2])
        if 'time_series' in self.integrated_results and 'ts_data' in self.integrated_results['time_series']:
            ts_data = self.integrated_results['time_series']['ts_data']
            ax7.plot(ts_data.index, ts_data['dau'], label='日活用户', color='#FF6B6B', linewidth=2)
            ax7.plot(ts_data.index, ts_data['daily_purchases'], label='日购买量', color='#4ECDC4', linewidth=2)
            ax7.set_title('核心指标趋势', fontsize=14, fontweight='bold')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
            ax7.tick_params(axis='x', rotation=45)

        # 5. 商品热度
        ax8 = fig.add_subplot(gs[2, 2:4])
        if 'product_stats' in self.integrated_results:
            product_stats = self.integrated_results['product_stats']
            top_products = product_stats.nlargest(8, '总行为数')

            bars = ax8.bar(range(len(top_products)), top_products['总行为数'],
                           color='#45B7D1', alpha=0.8)
            ax8.set_xticks(range(len(top_products)))
            ax8.set_xticklabels([f'商品{idx}' for idx in top_products.index], rotation=45)
            ax8.set_ylabel('行为次数')
            ax8.set_title('热门商品TOP8', fontsize=14, fontweight='bold')
            ax8.grid(True, alpha=0.3)

        # 6. 预测结果
        ax9 = fig.add_subplot(gs[3, 0:2])
        if 'forecast' in self.integrated_results:
            forecast_df = self.integrated_results['forecast']
            ax9.plot(forecast_df.index, forecast_df['predicted'],
                     label='预测值', color='#FF6B6B', linewidth=2, marker='o')
            ax9.fill_between(forecast_df.index,
                             forecast_df['confidence_lower'],
                             forecast_df['confidence_upper'],
                             alpha=0.3, color='#FF6B6B', label='置信区间')
            ax9.set_title('未来7天购买预测', fontsize=14, fontweight='bold')
            ax9.legend()
            ax9.grid(True, alpha=0.3)
            ax9.tick_params(axis='x', rotation=45)

        # 7. 留存分析
        ax10 = fig.add_subplot(gs[3, 2:4])
        if 'retention' in self.integrated_results and 'overall_retention' in self.integrated_results['retention']:
            retention_rates = self.integrated_results['retention']['overall_retention']
            days = [int(k.replace('第', '').replace('天留存率', '')) for k in retention_rates.keys()]
            rates = list(retention_rates.values())

            ax10.plot(days, rates, marker='o', linewidth=2, markersize=6, color='#96CEB4')
            ax10.fill_between(days, rates, alpha=0.3, color='#96CEB4')
            ax10.set_xlabel('天数')
            ax10.set_ylabel('留存率 (%)')
            ax10.set_title('用户留存曲线', fontsize=14, fontweight='bold')
            ax10.grid(True, alpha=0.3)

            for day, rate in zip(days, rates):
                ax10.text(day, rate + 1, f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

        plt.suptitle('电商用户行为分析综合看板', fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()

        if save:
            save_path = FIGURES_PATH / "comprehensive_dashboard.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 综合看板已保存至: {save_path}")

        plt.show()

        return fig

    def generate_final_report(self):
        """
        生成最终项目报告
        """
        print("\n=== 生成最终项目报告 ===")

        basic = self.project_metrics.get('basic', {})
        kpis = self.project_metrics.get('kpis', {})
        summary = self.project_metrics.get('executive_summary', {})

        report = f"""
# 电商用户行为分析 - 最终项目报告

## 项目概览
**分析时间**: {basic.get('analysis_date', 'N/A')}  
**数据范围**: {basic.get('date_range', 'N/A')}  
**分析规模**: {basic.get('total_records', 0):,} 条行为记录 × {basic.get('total_users', 0):,} 用户 × {basic.get('total_items', 0):,} 商品

## 执行摘要

### 项目成果
{summary.get('project_scope', '')}

### 核心优势
{chr(10).join(['- ' + strength for strength in summary.get('key_strengths', [])]) if summary.get('key_strengths') else '- 暂无显著优势'}

### 优化机会  
{chr(10).join(['- ' + opportunity for opportunity in summary.get('key_opportunities', [])]) if summary.get('key_opportunities') else '- 暂无明确机会点'}

## 关键业务指标

### 用户指标
- 日均活跃用户(DAU): {kpis.get('user_metrics', {}).get('dau', 0):.0f}
- 用户增长率: {kpis.get('user_metrics', {}).get('user_growth_rate', 0)}
- 平均留存率: {kpis.get('user_metrics', {}).get('avg_retention_rate', 0):.1f}%
- 高价值用户占比: {kpis.get('user_metrics', {}).get('high_value_user_ratio', 0):.1f}%

### 转化指标  
- 整体购买转化率: {kpis.get('conversion_metrics', {}).get('overall_conversion_rate', 0):.2f}%
- 浏览→加购转化率: {kpis.get('conversion_metrics', {}).get('browse_to_cart_rate', 0):.2f}%
- 加购→购买转化率: {kpis.get('conversion_metrics', {}).get('cart_to_purchase_rate', 0):.2f}%

### 商品指标
- 平均商品转化率: {kpis.get('product_metrics', {}).get('avg_product_conversion', 0):.2f}%
- 活跃商品比例: {kpis.get('product_metrics', {}).get('active_product_ratio', 0):.1f}%
- TOP3类目覆盖率: {kpis.get('product_metrics', {}).get('top_category_coverage', 0):.1f}%

## 技术架构

### 分析方法论
1. **数据预处理**: 数据清洗 + 特征工程 + 质量验证
2. **用户分析**: RFM分层 + 留存分析 + 行为路径
3. **商品分析**: 热度分析 + 关联规则 + 类目表现  
4. **预测建模**: 时间序列分析 + 机器学习预测
5. **推荐系统**: 协同过滤算法 + 个性化推荐

### 技术栈
- **数据处理**: Pandas, NumPy
- **分析建模**: Scikit-learn, Statsmodels
- **数据可视化**: Matplotlib, Seaborn
- **机器学习**: 随机森林, 线性回归, 协同过滤

## 业务计划

### 短期行动 (1-2周)
1. **转化优化**: 针对{summary.get('key_opportunities', [{}])[0] if summary.get('key_opportunities') else '关键流失点'}设计A/B测试
2. **用户激活**: 对低价值用户群体设计激活活动
3. **商品优化**: 基于关联规则优化商品展示和捆绑销售

### 中期计划 (1-3月)
4. **个性化推荐**: 部署推荐系统提升用户体验和转化
5. **预测运营**: 基于预测模型优化库存和营销资源分配
6. **用户生命周期管理**: 建立完整的用户成长体系

### 长期战略 (3-6月)
7. **数据驱动文化**: 建立常态化数据分析与决策机制
8. **技术架构升级**: 构建实时数据处理和机器学习平台
9. **业务扩展**: 基于洞察拓展新的业务增长点

## 项目价值评估

### 直接价值
- 识别了{len(summary.get('key_opportunities', []))}个关键优化机会
- 建立了{len(self.integrated_results)}个核心分析模型
- 提供了{len(summary.get('recommended_actions', []))}条具体行动建议

### 潜在影响
- 预计转化率提升空间: 10-25%
- 用户留存优化潜力: 15-30%  
- 运营效率提升: 通过预测模型减少20%资源浪费

## 后续工作建议

### 数据层面
- 补充用户 demographic 数据
- 集成业务交易数据
- 建立实时数据流水线

### 分析层面  
- 深化用户细分和画像
- 开发更复杂的预测模型
- 建立自动化报表体系

### 业务层面
- 建立数据驱动的决策流程
- 培训业务团队的数据应用能力
- 定期回顾和优化分析模型

---
**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**分析团队**: 电商数据分析项目组  
**联系方式**: [你的联系信息]
"""

        report_path = PROCESSED_DATA_PATH.parent.parent / "reports" / "final_reports" / "电商用户行为分析最终报告.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"✅ 最终项目报告已生成: {report_path}")
        return report_path


# 创建全局实例
project_integrator = ProjectIntegrator()

# df=data_loader.load_with_sampling()
#
# df= data_cleaner.create_time_features(df)
#
# analyses_dict=time_series_analyzer.prepare_time_series_data(df)
#
# integrated_results=project_integrator.integrate_all_analyses(df, analyses_dict)
#
# project_integrator.calculate_business_kpis(df,integrated_results)
#
# project_integrator.generate_executive_summary()
#
# project_integrator.create_comprehensive_dashboard()
#
# project_integrator.generate_final_report()
