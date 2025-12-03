"""商品分析模块
分析商品热度，表现和关联关系"""
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx


from src.config import FIGURES_PATH
from src.data_cleaner import data_cleaner
from src.data_loader import data_loader
plt.rcParams['font.sans-serif'] = ['SimHei']  

plt.rcParams['axes.unicode_minus'] = False

class ProductAnalyzer:
    def __init__(self):
        self.top_n=20

    def analyze_product_popularity(self,df):
        """分析商品热度和表现"""

        print("===商品热度分析===")
        #商品行为统计
        product_stats=df.groupby('item_id').agg({
            'behavior_type':'count',
            'user_id':'count',
            'category_id':'first'
        }).rename(columns={
            'behavior_type':'总行为数',
            'user_id':'触达用户数'
        })
        #计算购买行为
        purchaes_stats=df[df['behavior_type']=='buy'].groupby('item_id').agg({
            'user_id':'count',
            'behavior_type':'count'
        }).rename(columns={
            'user_id':'购买用户数',
            'behavior_type':'购买次数'
        })
        #合并数据
        product_stats=product_stats.merge(purchaes_stats,on='item_id',how='left').fillna(0)

        #计算转化率
        product_stats['购买转化率']=product_stats['购买次数']/product_stats['总行为数']*100
        #添加排名
        product_stats['热度排名']=product_stats['总行为数'].rank(ascending=False)
        product_stats['购买排名']=product_stats['购买次数'].rank(ascending=False)
        product_stats['转化排名']=product_stats['购买转化率'].rank(ascending=False)
        print(f"总商品数:{len(product_stats):,}")
        print(f"有购买行为的商品:{len(product_stats[product_stats['购买次数']>0]):,}")
        return product_stats
    def analyze_category_performance(self,df,product_stats):
        """分析类目表现"""
        print("\n===商品类目分析===")

        #类目统计
        category_stats=df.groupby('category_id').agg({
            'item_id':'nunique',
            'user_id':'nunique',
            'behavior_type':'count'
        }).rename(columns={
            'item_id':'商品数',
            'user_id':'触达用户数',
            'behavior_type':'总行为数'
        })
        #购买统计
        category_purchase=df[df['behavior_type']=='buy'].groupby('category_id').agg({
            'user_id':'count',
            'item_id':'nunique'
        }).rename(columns={
            'user_id':'购买次数',
            'item_id':'被购商品数'
        })
        category_stats=category_stats.merge(category_purchase,on='category_id',how='left').fillna(0)
        #计算类目指标
        category_stats['平均商品热度'] = category_stats['总行为数'] / category_stats['商品数']
        category_stats['购买转化率'] = category_stats['购买次数'] / category_stats['总行为数'] * 100
        category_stats['商品动销率'] = category_stats['被购商品数'] / category_stats['商品数'] * 100
        print(f"总类目数:{len(category_stats):,}")
        print("TOP5热销类目")
        top_category=category_stats.nlargest(5,"总行为数")
        for i,(category,row) in enumerate(top_category.iterrows(),1):
            print(f"{i}.类目{category}:{row['总行为数']:,}次行为")
        return category_stats

    def find_product_associations(self,df,min_support=0.01):
        """发现商品关联规则（购物篮分析）"""

        print("\n===商品关联规则分析===")
        #获取用户的购买序列（只考虑同一session的购买）
        user_purchases=df[df['behavior_type']=='buy'].groupby(['user_id','date'])['item_id'].apply(list)
        #计算商品共现频率
        cooccurrence=Counter()

        for items in user_purchases:
            if len(items)>=2:  #只考虑购买两个商品以上的情况
                for pair in combinations(set(items),2):
                    cooccurrence[pair]+=1

        #计算支持度
        total_sessions=len(user_purchases)
        associations_rules=[]

        for (item1, item2), count in cooccurrence.most_common(50):
            support=count/total_sessions
            if support>=min_support:
                associations_rules.append({
                    '商品A':item1,
                    '商品B':item2,
                    '共现次数':count,
                    '支持度':support
                })

        association_df=pd.DataFrame(associations_rules)

        if not association_df.empty:
            print(f"找到{len(association_df)}条强关联规则(支持度>={min_support*100}%)")
            print("TOP10关联规则:")
            print(association_df.head(10).round(4))
        else:
            print("未找到强关联规则，尝试降低min_suporrt参数")

        return association_df

    def create_product_visualization(self,product_stats,category_stats,association_df,save=True):
        """创建商品分析可视化"""
        print("正在生成商品分析图表...")
        fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(18,12))
        #图表1:商品热度top20
        top_products=product_stats.nlargest(20,'总行为数')
        bars1=ax1.barh(range(len(top_products)),top_products['总行为数'],
                       color='#FF6B6B',alpha=0.8)
        ax1.set_yticks(range(len(top_products)))
        ax1.set_yticklabels([f'商品{idx}' for idx in top_products.index])
        ax1.set_xlabel('行为次数')
        ax1.set_title('热门商品TOP20（按行为次数）', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()

        # 添加数据标签
        for i, (bar, count) in enumerate(zip(bars1, top_products['总行为数'])):
            ax1.text(bar.get_width() + max(top_products['总行为数']) * 0.01,
                     bar.get_y() + bar.get_height() / 2,
                     f'{count:,}', va='center', fontweight='bold')

        # 图表2: 类目行为分布
        top_categories = category_stats.nlargest(10, '总行为数')
        colors2 = plt.cm.Set3(np.linspace(0, 1, len(top_categories)))
        wedges2, texts2, autotexts2 = ax2.pie(top_categories['总行为数'],
                                              labels=[f'类目{idx}' for idx in top_categories.index],
                                              autopct='%1.1f%%', colors=colors2, startangle=90)
        ax2.set_title('TOP10类目行为分布', fontsize=14, fontweight='bold')

        # 图表3: 商品转化率分布
        products_with_purchase = product_stats[product_stats['购买次数'] > 0]
        if len(products_with_purchase) > 0:
            conversion_rates = products_with_purchase['购买转化率']
            ax3.hist(conversion_rates, bins=30, color='#4ECDC4', alpha=0.7, edgecolor='black')
            ax3.set_xlabel('购买转化率 (%)')
            ax3.set_ylabel('商品数量')
            ax3.set_title('商品购买转化率分布', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)

            # 添加统计信息
            avg_conversion = conversion_rates.mean()
            ax3.axvline(avg_conversion, color='red', linestyle='--', linewidth=2,
                        label=f'平均转化率: {avg_conversion:.2f}%')
            ax3.legend()

        # 图表4: 商品关联网络（简化版）
        if not association_df.empty:
            # 创建关联网络
            G = nx.Graph()

            # 添加节点和边（只取前10条强关联）
            top_associations = association_df.head(10)
            for _, row in top_associations.iterrows():
                G.add_edge(f"商品{row['商品A']}", f"商品{row['商品B']}",
                           weight=row['支持度'])

            # 绘制网络图
            pos = nx.spring_layout(G, seed=42)
            node_sizes = [300 + G.degree(node) * 100 for node in G.nodes()]
            edge_widths = [G[u][v]['weight'] * 100 for u, v in G.edges()]

            nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                                   node_color='lightblue', alpha=0.7, ax=ax4)
            nx.draw_networkx_edges(G, pos, width=edge_widths,
                                   alpha=0.5, edge_color='gray', ax=ax4)
            nx.draw_networkx_labels(G, pos, font_size=8, ax=ax4)

            ax4.set_title('商品关联网络图', fontsize=14, fontweight='bold')
            ax4.axis('off')

        plt.tight_layout()

        if save:
            save_path = FIGURES_PATH / "product_analysis.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 商品分析图表已保存至: {save_path}")

        plt.show()

        return fig

#
# 创建全局实例
product_analyzer = ProductAnalyzer()
#
# # 1. 加载原始数据（关键：所有方法都依赖原始df）
# df = data_loader.load_with_sampling()  # 先加载原始数据，再传递给各个方法
# date=data_cleaner.creat_time_features(df)
# # 2. 分析商品热度（参数：原始df）
# product_stats = product_analyzer.analyze_product_popularity(df)
#
# # 3. 分析类目表现（参数：原始df + 商品统计结果）
# category_stats = product_analyzer.analyze_categeory_performance(df, product_stats)
#
# # 4. 发现商品关联规则（参数：原始df，可选调整min_support）
# association_df = product_analyzer.find_product_associations(date, min_support=0.005)  # 降低支持度更容易找到关联
#
# # 5. 生成可视化（参数：商品统计 + 类目统计 + 关联规则）
# fig = product_analyzer.create_product_visualization(product_stats, category_stats, association_df, save=True)
#

