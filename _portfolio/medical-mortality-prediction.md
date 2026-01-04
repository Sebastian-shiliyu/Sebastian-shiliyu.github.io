---
title: "医疗数据死亡率预测分析"
collection: portfolio
type: "Data Analysis"
permalink: /portfolio/medical-mortality-prediction
date: 2026-01-04
excerpt: "基于ICU患者临床数据的探索性分析与死亡率预测模型构建，处理不平衡数据并解释模型决策"
header:
  teaser: /images/portfolio/medical-mortality-prediction/target_distribution.png
tags:
  - 医疗数据分析
  - 机器学习
  - 预测建模
  - 数据预处理
  - 不平衡学习
  - 可解释AI
tech_stack:
  - name: Python
  - name: Pandas
  - name: Scikit-learn
  - name: TensorFlow/Keras
  - name: Matplotlib/Seaborn
---

## 项目背景

本项目基于ICU（重症监护室）患者的临床监测数据，构建死亡率预测模型。数据集包含13,258条患者记录，7个特征变量（包括年龄、5项实验室指标最小值/极差、以及住院死亡标志）。项目核心挑战包括：

1. **数据质量问题**：超过30%的缺失值、重复记录和异常值
2. **类别不平衡**：死亡率仅5.88%，存在严重的不平衡问题
3. **可解释性需求**：医疗场景需要理解模型决策依据
4. **临床实用性**：模型需要具备实际的预测价值

通过系统的数据探索、预处理、特征工程和多种建模策略，最终构建了一个具有良好性能且可解释的预测系统。

## 核心实现

### 数据预处理与清洗

```python
# 核心预处理逻辑 - 处理缺失值、重复值和异常值
def preprocess_data(df):
    """
    数据清洗与预处理函数
    1. 使用中位数填充缺失值
    2. 删除重复行
    3. 使用IQR方法检测并处理异常值
    """
    processed_df = df.copy()
    
    # 缺失值处理
    numeric_columns = processed_df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        median_val = processed_df[col].median()
        processed_df[col].fillna(median_val, inplace=True)
    
    # 删除重复行
    processed_df.drop_duplicates(inplace=True)
    
    # 异常值检测与处理（IQR方法）
    for col in numeric_columns:
        Q1 = processed_df[col].quantile(0.25)
        Q3 = processed_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 将异常值截断到边界
        processed_df[col] = processed_df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return processed_df
```

### 探索性数据分析（EDA）

```python
# 特征分布可视化
def plot_numeric_distributions(df):
    """
    绘制数值变量的分布图：直方图、KDE密度图和小提琴图
    用于理解数据分布特征和识别潜在问题
    """
    numeric_cols = ['age_month', 'lab_5237_min', 'lab_5227_min', 
                    'lab_5225_range', 'lab_5235_max', 'lab_5257_min']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols):
        # 绘制直方图
        axes[idx].hist(df[col], bins=30, alpha=0.7, 
                       color='steelblue', edgecolor='black')
        axes[idx].set_title(f'{col} 分布')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('频数')
        
        # 添加描述性统计文本
        mean_val = df[col].mean()
        median_val = df[col].median()
        axes[idx].axvline(mean_val, color='red', linestyle='--', 
                         label=f'均值: {mean_val:.2f}')
        axes[idx].axvline(median_val, color='green', linestyle='-', 
                         label=f'中位数: {median_val:.2f}')
        axes[idx].legend()
    
    plt.tight_layout()
    plt.show()

# 相关系数分析
def analyze_correlations(df):
    """
    计算特征相关性矩阵并绘制热力图
    """
    correlation_matrix = df.corr()
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                center=0, square=True, linewidths=0.5)
    plt.title('特征相关系数矩阵')
    plt.tight_layout()
    plt.show()
    
    return correlation_matrix
```

### 机器学习建模

```python
# 多层感知机(MLP)模型构建
def build_mlp(input_dim):
    """
    构建用于死亡率预测的多层感知机
    架构：输入层 → 批量归一化 → 256单元ReLU → Dropout(0.3) 
    → 64单元ReLU → Dropout(0.2) → Sigmoid输出
    """
    model = Sequential([
        BatchNormalization(input_shape=(input_dim,)),
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        BatchNormalization(),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    # 使用Adam优化器和二元交叉熵损失
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy', 
                  metrics=['accuracy', AUC(name='auc')])
    
    return model

# 模型训练与评估
def train_and_evaluate(model, X_train, y_train, X_test, y_test, 
                      epochs=50, batch_size=32):
    """
    训练模型并评估性能
    包含早停机制和ROC曲线绘制
    """
    # 早停回调防止过拟合
    early_stop = EarlyStopping(monitor='val_loss', patience=10, 
                              restore_best_weights=True)
    
    # 训练模型
    history = model.fit(X_train, y_train, epochs=epochs, 
                       batch_size=batch_size, validation_split=0.2,
                       callbacks=[early_stop], verbose=0)
    
    # 预测与评估
    y_pred_proba = model.predict(X_test).ravel()
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC曲线 (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('ROC曲线')
    plt.legend(loc="lower right")
    plt.show()
    
    return model, history, roc_auc, y_pred_proba
```

### 不平衡数据处理与模型解释

```python
# 使用SMOTE进行过采样处理不平衡数据
from imblearn.over_sampling import SMOTE

def balance_data(X_train, y_train):
    """
    使用SMOTE合成少数类过采样技术平衡训练数据
    """
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    return X_balanced, y_balanced

# SHAP模型解释
import shap

def explain_model(model, X_test, feature_names):
    """
    使用SHAP值解释模型预测
    生成特征重要性汇总图
    """
    # 创建SHAP解释器
    explainer = shap.KernelExplainer(model.predict, X_test[:100])
    shap_values = explainer.shap_values(X_test[:100])
    
    # 绘制SHAP汇总图
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names, 
                      plot_type="bar", show=False)
    plt.title('SHAP特征重要性分析')
    plt.tight_layout()
    plt.show()
    
    return shap_values
```

## 分析结果

### 数据质量分析

目标变量分布显示明显的类别不平衡（死亡率仅5.88%），这需要在建模时特别处理。

![目标变量分布](/images/portfolio/medical-mortality-prediction/target_distribution.png)

缺失值分析显示所有实验室指标缺失率均超过30%，需要合理的填充策略。

![缺失值热力图](/images/portfolio/medical-mortality-prediction/missing_values_heatmap.png)


相关系数矩阵揭示了特征间的相关性结构，为特征选择提供依据。

![相关系数](/images/portfolio/medical-mortality-prediction/correlation_heatmap.png)

### 模型性能

初始模型在原始不平衡数据上达到AUC 0.75，但存在对多数类的过度拟合。

![ROC曲线](/images/portfolio/medical-mortality-prediction/roc_curve.png)



