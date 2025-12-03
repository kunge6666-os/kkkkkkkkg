import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from src.config import FIGURES_PATH
from src.data_loader import data_loader


class SimpleRecommender:
    def __init__(self):
        self.user_item_matrix = None
        self.item_similarity = None
        self.user_similarity = None

    def create_user_item_matrix(self,df):
        """创建用户_商品交互矩阵"""
        print("===构建用户_商品矩阵===")

        #为不同行为类型分配权重
        behavior_weights={
            'pv': 1,    #浏览权重
            'cart': 3,  #加购权重
            'fav':4,    #收藏权重
            'buy':5     #购买权重
        }
        #计算用户对商品的加权兴趣分数
        df['weight']=df['behavior_type'].map(behavior_weights)
        user_item_scores=df.groupby(['user_id','item_id'])['weight'].sum().reset_index()

        #创建用户_商品矩阵
        self.user_item_matrix=user_item_scores.pivot(index='user_id',
                                                     columns='item_id',
                                                     values='weight'
        ).fillna(0)
        print(f"用户_商品矩阵形状：{self.user_item_matrix.shape}")
        print(f"稀疏度:{(self.user_item_matrix==0).sum().sum()/(self.user_item_matrix.shape[0]*self.user_item_matrix.shape[1])*100:.2f}%")
        return self.user_item_matrix
    def calculate_item_similarity(self):
        """计算商品相似度矩阵（基于协同过滤）"""
        print("计算产品相似度...")

        #使用余弦相似度计算商品相似度
        self.item_similarity=cosine_similarity(self.user_item_matrix.T)
        self.item_similarity_df=pd.DataFrame(
            self.item_similarity,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )
        print("商品相似度矩阵计算完成")
        return self.item_similarity_df

    def calculate_user_similarity(self):
        """计算用户相似度矩阵"""
        #使用余弦相似度计算用户相似度
        self.user_similarity=cosine_similarity(self.user_item_matrix)
        self.user_similarity_df=pd.DataFrame(
            self.user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        print("用户相似度矩阵计算完成")
        return self.user_similarity_df
    def recommend_for_user(self, user_id,n_recommendations=10,method='item_based'):
        """为用户生成推荐"""
        if method == 'item_based':
            return self._item_based_recommendation(user_id,n_recommendations)
        elif method == 'user_based':
            return self._user_based_recommendation(user_id,n_recommendations)
        else:
            raise ValueError("方法应该是'item_based' or 'user_based'")


    def _item_based_recommendation(self,user_id,n_recommendations):
        """基于商品的协同过滤推荐"""
        if user_id not in self.user_item_matrix.index:
            return f"用户{user_id}不在数据集中"
        #获取用户的历史交互商品
        user_interactions=self.user_item_matrix.loc[user_id]
        interacted_items= user_interactions.loc[user_interactions>0].index
        if len(interacted_items)==0:
            return "用户没有历史行为，无法生成推荐"

        #计算推荐分数
        recommendation_scores={}

        for item in self.user_item_matrix.columns:
            if item not in interacted_items:#只推荐未交互的商品
                #计算与用户历史交互商品的相似度加权和
                similarity_scores=[]
                weights=[]

                for interacted_item in interacted_items:
                    similarity=self.item_similarity_df.loc[item,interacted_item]
                    user_interest=user_interactions[interacted_item]

                    similarity_scores.append(similarity*user_interest)
                    weights.append(user_interest)
                if weights:
                    recommendation_scores[item]=sum(similarity_scores)/sum(weights)
                    #返回TopN推荐
                    top_recommendations=sorted(recommendation_scores.items(),key=lambda x:x[1],reverse=True)[:n_recommendations]
                    return pd.DataFrame(top_recommendations,columns=['商品ID','推荐分数'])

    def _user_based_recommendation(self,user_id,n_recommendations):
        """基于用户的协同过滤"""
        if user_id not in self.user_item_matrix.index:
            return f"用户{user_id}不在数据集中"
        #找到相似用户
        user_similarities=self.user_item_matrix.loc[user_id]
        similar_users=user_similarities.drop(user_id).nlargest(10)#Top10相似用户

        #获取目标用户的交互史
        target_user_interactions=set(
            self.user_item_matrix.loc[user_id][self.user_item_matrix>0].index

        )
        #计算推荐分数
        recommendation_scores={}

        for item in self.user_item_matrix.columns:
            if item not in target_user_interactions:
                score=0
                total_similarity=0
                for similar_user,similarity in similar_users.items():
                    user_rating=self.user_item_matrix.loc[similar_user,item]
                    score+=similarity*user_rating
                    total_similarity+=abs(similarity)


                if total_similarity>0:
                    recommendation_scores[item]=score/total_similarity

        #返回TopN推荐
        top_recommendations=sorted(recommendation_scores.items(),key=lambda x:x[1],reverse=True)[:n_recommendations]
        return pd.DataFrame(top_recommendations,columns=['商品ID','推荐分数'])

    def evaluate_recommendation(self,test_ratio=0.2):
        """简单评估推荐效果（训练集—测试集分割）"""
        #简化版评估：计算覆盖率
        all_items=set(self.user_item_matrix.columns)
        recommended_items=set()

        #为部分用户生成推荐并统计覆盖商品
        sample_users=self.user_item_matrix.sample(min(100,len(self.user_item_matrix))).index

        for user in sample_users:
            recommendations=self.recommend_for_user(user,10,'item_based')

            if isinstance(recommendations,pd.DataFrame):
                recommended_items.update(recommendations['商品ID'].tolist())
        coverage=len(recommended_items)/len(all_items)*100
        print(f"推荐覆盖率：{coverage:.2f}% ({len(recommended_items)}/{len(all_items)}商品)")
        return coverage

    def create_recommendation_demo(self,user_ids=None,save=True):
        """创建推荐演示"""
        print("生成推荐演示...")
        # 解决中文显示乱码
        plt.rcParams['font.sans-serif'] = ['SimHei'] if plt.get_backend() != 'MacOSX' else ['Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        if user_ids is None:
            #选择一些有代表性的用户
            active_users=self.user_item_matrix.sum(axis=1).nlargest(5).index
            new_users=self.user_item_matrix.sum(axis=1).nsmallest(2).index
            user_ids=list(active_users)[:3]+list(new_users)[:1]

        fig,axes=plt.subplots(2,2,figsize=(10,10))
        axes=axes.flatten()

        for i,user_id in enumerate(user_ids[:4]):
            ax=axes[i]

            #生成推荐
            recommendations=self.recommend_for_user(user_id,5,'item_based')
            if isinstance(recommendations,str):
                ax.text(0.5,0.5,recommendations,ha='center',va='center',
                        transform=ax.transAxes,fontsize=11)
                ax.set_title(f'用户{user_id}推荐结果',fontsize=12)
            else:
                #显示推荐结果
                bars=ax.barh(range(len(recommendations)),
                             recommendations['推荐分数'],
                             color='#4ECDC4',alpha=0.8)
                ax.set_yticks(range(len(recommendations)))
                ax.set_yticklabels([f'商品{idx}'for idx in recommendations['商品ID']])
                ax.set_xlabel('推荐分数')
                ax.set_title(f'用户{user_id}的top5推荐',fontsize=12)
                ax.invert_yaxis()

                #添加标签
                for j,(bar,score) in enumerate(zip(bars,recommendations['推荐分数'])):
                    ax.text(bar.get_width()+0.01,bar.get_y()+bar.get_height()/2,
                            f'{score:.3f}',va='center',fontsize=10)
        plt.tight_layout()

        if save:
            save_path = FIGURES_PATH / "recommendation_demo.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 推荐演示图表已保存至: {save_path}")

        plt.show()

#创建全局实例
recommender = SimpleRecommender()
# recommender.create_user_item_matrix(data_loader.load_random())
# recommender.calculate_item_similarity()
#
# # 4. 生成并显示推荐图表
#
# recommender.create_recommendation_demo(
#     user_ids=None,
#     save=True  # 保存图表（True/False）
# )











