import numpy as np
from sklearn.metrics import mean_squared_error
import random

# 1. 读取训练数据 train.txt
train_data = {}
with open('train.txt', 'r') as f_train:
    for line in f_train:
        user_id, num_items = line.strip().split('|')
        num_items = int(num_items)
        item_ratings = {}
        for _ in range(num_items):
            item_id, score = map(int, f_train.readline().strip().split())
            item_ratings[item_id] = score
        train_data[user_id] = item_ratings

# 2. 拆分训练集和验证集
validation_data = {}
validation_ratio = 0.1  # 验证集占比
for user_id, item_ratings in train_data.items():
    num_items = len(item_ratings)
    num_validation = int(num_items * validation_ratio)
    validation_items = random.sample(item_ratings.items(), num_validation)
    train_items = [(item_id, score) for item_id, score in item_ratings.items() if (item_id, score) not in validation_items]

    train_data[user_id] = dict(train_items)
    validation_data[user_id] = dict(validation_items)

# 3. Matrix Factorization 推荐算法

# 创建用户-物品矩阵
def create_user_item_matrix(data):
    users = sorted(data.keys())
    items = set()
    for item_ratings in data.values():
        items.update(item_ratings.keys())
    items = sorted(items)
    
    user_item_matrix = np.zeros((len(users), len(items)))
    for i, user_id in enumerate(users):
        item_ratings = data[user_id]
        for j, item_id in enumerate(items):
            rating = item_ratings.get(item_id, 0)
            user_item_matrix[i, j] = rating
    
    return user_item_matrix, users, items

# 训练模型
def matrix_factorization(train_matrix, n_factors=10, n_iterations=10, learning_rate=0.01, reg_param=0.01, random_seed=42):
    np.random.seed(random_seed)
    
    n_users, n_items = train_matrix.shape
    
    # 初始化用户矩阵和物品矩阵
    user_matrix = np.random.normal(scale=1./n_factors, size=(n_users, n_factors))
    item_matrix = np.random.normal(scale=1./n_factors, size=(n_items, n_factors))
    
    for _ in range(n_iterations):
        for i in range(n_users):
            for j in range(n_items):
                if train_matrix[i, j] > 0:
                    error = train_matrix[i, j] - np.dot(user_matrix[i, :], item_matrix[j, :].T)
                    user_matrix[i, :] += learning_rate * (2 * error * item_matrix[j, :] - reg_param * user_matrix[i, :])
                    item_matrix[j, :] += learning_rate * (2 * error * user_matrix[i, :] - reg_param * item_matrix[j, :])
    
    return user_matrix, item_matrix

# 创建训练集和验证集的用户-物品矩阵
train_matrix, train_users, train_items = create_user_item_matrix(train_data)
validation_matrix, _, _ = create_user_item_matrix(validation_data)

# 训练MF模型
n_factors = 10  # 因子数量
n_iterations = 10  # 迭代次数
learning_rate = 0.01  # 学习率
reg_param = 0.01  # 正则化参数

user_matrix, item_matrix = matrix_factorization(train_matrix, n_factors, n_iterations, learning_rate, reg_param)

# 4. 计算验证集的 RMSE 评价指标
def calculate_rmse(predicted_matrix, true_matrix):
    non_zero_indices = true_matrix.nonzero()
    true_values = true_matrix[non_zero_indices]
    predicted_values = predicted_matrix[non_zero_indices]
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    return rmse

validation_predicted = np.dot(user_matrix, item_matrix.T)
validation_rmse = calculate_rmse(validation_predicted, validation_matrix)
print(f"RMSE (Validation): {validation_rmse}")

# 5. 进行测试集预测，并输出结果到 output.txt
with open('test.txt', 'r') as f_test, open('output2.txt', 'w') as f_output:
    for line in f_test:
        user_id, num_items = line.strip().split('|')
        num_items = int(num_items)
        f_output.write(f"{user_id}|{num_items}\n")
        for _ in range(num_items):
            item_id = int(f_test.readline().strip())
            user_index = train_users.index(user_id)
            item_index = train_items.index(item_id)
            predicted_score = np.dot(user_matrix[user_index, :], item_matrix[item_index, :].T)
            f_output.write(f"{item_id}\t{predicted_score}\n")
print("finish")