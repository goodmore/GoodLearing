# 导入必要的库
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import numpy as np

# 示例训练数据：每行是一条邮件内容，后面是标签（'spam' 或 'ham'）
train_data = [
    ("免费获取 iPhone 大奖！点击链接", "spam"),
    ("老板，下午三点开会，请准时参加", "ham"),
    ("恭喜您中奖了！立即领取您的奖金", "spam"),
    ("项目报告已发到您的邮箱，请查收", "ham"),
    ("限时特价，全场五折，仅限今天", "spam"),
    ("周末聚餐定在晚上七点，老地方", "ham")
]

# 1. 准备数据：将文本和标签分开
texts = [data[0] for data in train_data]  # 邮件文本列表
labels = [data[1] for data in train_data] # 对应标签列表

# 2. 创建并训练模型管道
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(texts, labels)

# 3. 准备新邮件进行预测
new_emails = [
    "免费领取优惠券，机会难得！",  # 预期为 spam
    "明天上午十点电话会议讨论预算"   # 预期为 ham
]

# 4. 进行预测
predictions = model.predict(new_emails)
prediction_proba = model.predict_proba(new_emails) # 获取预测概率

# 5. 输出结果（修复引号问题 + 动态匹配概率标签）
# 获取模型的类别顺序（避免硬编码索引）
class_names = model.classes_
for email, pred, proba in zip(new_emails, predictions, prediction_proba):
    # 修复引号嵌套问题：内层改用单引号，或外层用单引号
    print(f'邮件内容: "{email}"')
    print(f"  预测类别: {pred}")
    # 动态输出每个类别的概率（更健壮）
    for cls, prob in zip(class_names, proba):
        print(f"  属于'{cls}'的概率: {prob:.4f}")
    print("-" * 40)