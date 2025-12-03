"""预测模型模块"""
from multiprocessing.spawn import prepare
from sklearn.linear_model import LinearRegression

import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

from src.config import FIGURES_PATH
from src.data_cleaner import data_cleaner
from src.data_loader import data_loader
from src.time_series_analyzer import time_series_analyzer


class PredictiveModel:
    def __init__(self):
        self.models={}
        self.scaler=StandardScaler()
        self.feature_importance=None

    def prepare_features(self,time_series_data,target_column='daily_purchases',lag_days=7):
        """准备机器学习特征"""
        print("特征工程")
        features_df=time_series_data.copy()

        #创建滞后特征

        for lag in range(1,lag_days+1):
            features_df[f'{target_column}_lag_{lag}']=features_df[target_column].shift(lag)
            features_df[f'dau_lag_{lag}']=features_df['dau'].shift(lag)
            features_df[f'daily_actions_lag{lag}']=features_df['daily_actions'].shift(lag)

        #创建滚动统计特征
        features_df[f'{target_column}_rolling_mean_7']=features_df[target_column].rolling(7).mean()
        features_df[f'{target_column}_rolling_std_7']=features_df[target_column].rolling(7).std()

        #创建时间特征
        features_df.index = pd.to_datetime(features_df.index)
        features_df['day_of_week']=features_df.index.dayofweek
        features_df['day_of_month']=features_df.index.day
        features_df['is_weekend']=features_df['day_of_week'].isin([5,6]).astype(int)

        #删除包含NnN的行
        features_df=features_df.dropna()

        print(f"特征数量{len([col for col in features_df.columns if col != target_column])}")

        print(f"有效样本数:{len(features_df)}")
        return features_df

    def train_models(self, features_df, target_column='daily_purchases', test_size=0.2):
        """
        训练多个预测模型
        """
        print("\n=== 模型训练 ===")

        # 准备特征和目标变量
        feature_columns = [col for col in features_df.columns if col not in [target_column]]
        X = features_df[feature_columns]
        y=features_df[target_column]

        # 时间序列分割（保持时间顺序）
        tscv = TimeSeriesSplit(n_splits=5)

        # 初始化模型
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=150, random_state=42,max_depth=8,min_samples_split=10,min_samples_leaf=5,max_features='sqrt',  # 每次分裂用sqrt(特征数)：减少特征冗余，提升泛化能力
                bootstrap=True),
            'LinearRegression': LinearRegression()
        }

        model_results = {}

        for model_name, model in models.items():
            print(f"训练 {model_name}...")

            # 交叉验证
            cv_scores = []
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # 训练模型
                model.fit(X_train, y_train)

                # 预测
                y_pred = model.predict(X_test)
                y_pred = np.clip(y_pred, a_min=0, a_max=None)

                # 评估
                mae = mean_absolute_error(y_test, y_pred)
                cv_scores.append(mae)

            # 最终模型训练
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=False, random_state=42
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # 计算评估指标
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            model_results[model_name] = {
                'model': model,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'cv_mean_mae': np.mean(cv_scores),
                'cv_std_mae': np.std(cv_scores),
                'feature_importance': getattr(model, 'feature_importances_', None)
            }

            print(f"  {model_name} - MAE: {mae:.2f}, R²: {r2:.4f}")

        self.models = model_results
        return model_results

    def analyze_feature_importance(self,features_df,target_column='daily_purchases'):
        """分析特征重要性"""
        print("\n特征重要性分析")
        if'RandomForest' not in self.models:
            print("随机森林模型训练")
            return None

        rf_model=self.models['RandomForest']['model']
        feature_columns=[col for col in features_df.columns if col !=target_column]
        importance_df=pd.DataFrame({
            'feature':feature_columns,
            'importance':rf_model.feature_importances_
        }).sort_values(by='importance',ascending=False)

        print("top10的重要特征")
        for i,row in importance_df.head(10).iterrows():
            print(f"{row['feature']}:{row['importance']:.4f}")

        self.feature_importance=importance_df
        return importance_df

    def predict_future(self,features_df,target_column='daily_purchases',days=7):
        """预测未来值"""
        print(f"预测未来{days}天")

        if not self.models:
            print("没有训练好模型")
            return None

        #使用随机森林模型预测
        best_model_name=min(self.models.keys(),
                            key=lambda x: self.models[x]['mae'])#寻找最优模型
        best_model=self.models[best_model_name]['model']
        feature_column=[col for col in features_df.columns if col!=target_column]
        last_knowm_data=features_df[feature_column].iloc[-1:].copy()

        predictions=[]
        confidence_intervals=[]

        #递归预测
        current_features=last_knowm_data.copy()

        for day in range(days):
            #预测
            pred=best_model.predict(current_features)[0]
            predictions.append(pred)

            #更新特征（根据实际工程逻辑调整）
            #简化版：只会更新目标变量的滞后特征
            for lag in range(1,8):
                lag_col=f'{target_column}_lag_{lag}'
                if lag_col in current_features.columns:
                    if lag==1:
                        current_features[lag_col]=pred
                    else:
                        #需更强的逻辑来更新滞后特征
                        pass

            #简单的置信区间（基于历史误差）
            historical_errors=np.abs(self.models[best_model_name]['mae'])
            confidence_intervals.append((pred-historical_errors,pred+historical_errors))

        future_dates = pd.date_range(start=features_df.index[-1] + pd.Timedelta(days=1),
                                         periods=days, freq='D')

        forecast_df = pd.DataFrame({
            'date': future_dates,
            'predicted': predictions,
            'confidence_lower': [ci[0] for ci in confidence_intervals],
            'confidence_upper': [ci[1] for ci in confidence_intervals]
            }).set_index('date')

        print("未来预测结果:")
        print(forecast_df.round(2))

        return forecast_df

    def create_prediction_visualization(self, features_df, target_column='daily_purchases',
                                        forecast_df=None, save=True):
        """
        创建预测可视化
        """
        print("生成预测分析图表...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

        # 图表1: 模型性能比较
        model_names = list(self.models.keys())
        mae_scores = [self.models[name]['mae'] for name in model_names]

        bars = ax1.bar(model_names, mae_scores, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        ax1.set_ylabel('MAE (平均绝对误差)')
        ax1.set_title('模型性能比较', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        for bar, score in zip(bars, mae_scores):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(mae_scores) * 0.01,
                     f'{score:.2f}', ha='center', va='bottom', fontweight='bold')

        # 图表2: 特征重要性
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(10)
            y_pos = np.arange(len(top_features))

            ax2.barh(y_pos, top_features['importance'], color='#45B7D1', alpha=0.8)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(top_features['feature'])
            ax2.set_xlabel('重要性分数')
            ax2.set_title('TOP10特征重要性', fontsize=14, fontweight='bold')
            ax2.invert_yaxis()
            ax2.grid(True, alpha=0.3)

        # 图表3: 实际vs预测
        best_model_name = min(self.models.keys(), key=lambda x: self.models[x]['mae'])
        best_model = self.models[best_model_name]['model']

        feature_columns = [col for col in features_df.columns if col != target_column]
        X = features_df[feature_columns]
        y_true = features_df[target_column]

        # 使用最后20%数据展示预测效果
        split_idx = int(len(X) * 0.8)
        X_test = X.iloc[split_idx:]
        y_true_test = y_true.iloc[split_idx:]

        y_pred_test = best_model.predict(X_test)

        ax3.plot(y_true_test.index, y_true_test.values, label='实际值',
                 color='#FF6B6B', linewidth=2, marker='o')
        ax3.plot(y_true_test.index, y_pred_test, label='预测值',
                 color='#4ECDC4', linewidth=2, marker='s', linestyle='--')
        ax3.set_ylabel(target_column)
        ax3.set_title('实际值 vs 预测值', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 图表4: 未来预测
        if forecast_df is not None:
            # 显示最近30天的历史数据和未来预测
            recent_history = features_df[target_column].tail(30)

            ax4.plot(recent_history.index, recent_history.values,
                     label='历史数据', color='#45B7D1', linewidth=2)
            ax4.plot(forecast_df.index, forecast_df['predicted'],
                     label='预测值', color='#FF6B6B', linewidth=2, marker='o')

            # 绘制置信区间
            ax4.fill_between(forecast_df.index,
                             forecast_df['confidence_lower'],
                             forecast_df['confidence_upper'],
                             alpha=0.3, color='#FF6B6B', label='置信区间')

            ax4.set_ylabel(target_column)
            ax4.set_title('未来7天预测', fontsize=14, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            save_path = FIGURES_PATH / "predictive_analysis.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 预测分析图表已保存至: {save_path}")

        plt.show()

        return fig

# 创建全局实例
predictive_model = PredictiveModel()
# df=data_loader.load_with_sampling()
# date=data_cleaner.creat_time_features(df)
# time_series_data = time_series_analyzer.prepare_time_series_date(date)
# features_df=predictive_model.prepare_features(time_series_data)
# train_models=predictive_model.train_models(features_df)
# analyze_feature_importance=predictive_model.analyze_feature_importance(features_df)
# predict_future=predictive_model.predict_future(features_df)
# create_prediction_visualization=predictive_model.create_prediction_visualization(features_df)
