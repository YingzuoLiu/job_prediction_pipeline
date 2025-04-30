import pandas as pd
import numpy as np
import datetime
import os

def generate_user_data(num_users=10000):
    """生成模拟用户数据"""
    # 设置随机种子确保可复现
    np.random.seed(42)
    
    # 用户ID
    user_ids = [f"user_{i}" for i in range(1, num_users + 1)]
    
    # 随机生成一些重复账号（模拟一个用户有多个账号）
    duplicate_users = np.random.choice(user_ids, size=int(num_users * 0.1), replace=False)
    duplicate_mapping = {user: [f"{user}_alt_{i}" for i in range(1, np.random.randint(1, 4))] 
                         for user in duplicate_users}
    
    # 扩展用户列表，添加重复账号
    all_user_ids = user_ids.copy()
    for user, alts in duplicate_mapping.items():
        all_user_ids.extend(alts)
    
    # 生成用户基本信息
    data = {
        'user_id': all_user_ids,
        'registration_date': [
            datetime.datetime.now() - datetime.timedelta(days=np.random.randint(1, 1000))
            for _ in range(len(all_user_ids))
        ],
        'age': np.random.randint(18, 65, size=len(all_user_ids)),
        'gender': np.random.choice(['M', 'F', 'Other'], size=len(all_user_ids)),
        'location': np.random.choice(['北京', '上海', '广州', '深圳', '杭州', '成都', '重庆', '西安', '武汉', '南京'], 
                                   size=len(all_user_ids)),
        'education_level': np.random.choice(['高中', '大专', '本科', '硕士', '博士'], size=len(all_user_ids)),
    }
    
    # 模拟一些垃圾用户特征
    spam_indices = np.random.choice(range(len(all_user_ids)), size=int(len(all_user_ids) * 0.05), replace=False)
    for idx in spam_indices:
        data['registration_date'][idx] = datetime.datetime.now() - datetime.timedelta(hours=np.random.randint(1, 24))
        data['user_id'][idx] = f"bot_{np.random.randint(10000, 99999)}"
    
    # 创建DataFrame
    df_users = pd.DataFrame(data)
    
    # 添加一个是否垃圾用户的标记（实际场景中这个是需要模型预测的）
    df_users['is_spam'] = 0
    df_users.loc[spam_indices, 'is_spam'] = 1
    
    return df_users

def generate_user_activities(df_users, avg_activities_per_user=20):
    """生成用户活动记录"""
    activities = []
    
    # 活动类型
    activity_types = [
        'view_job_listing', 'search_job', 'update_resume', 'apply_job',  # 与找工作相关
        'read_article', 'view_profile', 'message_friend', 'update_status',  # 社交类活动
        'view_ad', 'click_ad'  # 广告相关
    ]
    
    # 模拟找工作特征
    # 大约30%的用户在找工作
    job_seekers = np.random.choice(df_users['user_id'].values, 
                                   size=int(len(df_users) * 0.3), 
                                   replace=False)
    
    for user_id in df_users['user_id']:
        # 确定该用户的活动数量
        if user_id in job_seekers:
            # 找工作的用户活动更多
            num_activities = np.random.randint(avg_activities_per_user, avg_activities_per_user * 2)
            
            # 找工作的用户活动偏向于找工作类活动
            weights = [0.3, 0.25, 0.2, 0.15, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025]
        else:
            num_activities = np.random.randint(1, avg_activities_per_user)
            
            # 非找工作用户的活动偏向社交
            weights = [0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.05, 0.05]
        
        is_spam = df_users[df_users['user_id'] == user_id]['is_spam'].values[0]
        
        # 垃圾用户的活动模式不同
        if is_spam:
            num_activities = np.random.randint(50, 200)  # 垃圾用户活动量更大
            weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # 随机分布
        
        # 生成活动
        for _ in range(num_activities):
            # 活动发生时间
            activity_time = datetime.datetime.now() - datetime.timedelta(
                days=np.random.randint(0, 60),
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60)
            )
            
            # 选择活动类型
            activity_type = np.random.choice(activity_types, p=weights)
            
            # 活动持续时间（分钟）
            duration = np.random.randint(1, 30)
            
            activities.append({
                'user_id': user_id,
                'timestamp': activity_time,
                'activity_type': activity_type,
                'duration': duration
            })
    
    # 创建活动DataFrame
    df_activities = pd.DataFrame(activities)
    
    # 添加目标变量：是否在找工作
    df_activities['is_job_seeking'] = df_activities['user_id'].apply(lambda x: 1 if x in job_seekers else 0)
    
    return df_activities

def generate_datasets():
    # 确保数据目录存在
    os.makedirs('data/raw', exist_ok=True)
    
    # 生成用户数据
    df_users = generate_user_data(num_users=10000)
    df_users.to_csv('data/raw/users.csv', index=False)
    print(f"已生成用户数据: {len(df_users)} 条记录")
    
    # 生成用户活动数据
    df_activities = generate_user_activities(df_users, avg_activities_per_user=20)
    df_activities.to_csv('data/raw/user_activities.csv', index=False)
    print(f"已生成用户活动数据: {len(df_activities)} 条记录")
    
    # 创建一个简单的数据概览
    with open('data/raw/README.md', 'w') as f:
        f.write('# 数据概览\n\n')
        f.write('## 用户数据\n')
        f.write(f'- 总用户数: {len(df_users)}\n')
        f.write(f'- 垃圾用户数: {df_users["is_spam"].sum()}\n\n')
        f.write('## 用户活动数据\n')
        f.write(f'- 总活动记录: {len(df_activities)}\n')
        f.write(f'- 找工作用户比例: {df_activities["is_job_seeking"].mean():.2%}\n')

if __name__ == "__main__":
    generate_datasets()
    print("示例数据生成完成！")