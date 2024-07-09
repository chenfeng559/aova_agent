import sqlite3
conn = sqlite3.connect('test.db')

c = conn.cursor()
# 创建表
c.execute('''
CREATE TABLE IF NOT EXISTS test (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time TEXT default (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')) NOT NULL,
    end_time TEXT default  (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')) ,
    location TEXT,
    description TEXT default  '' NOT NULL,
    participants TEXT
);
''')

# 插入数据 进行初始化

sql1 = """INSERT INTO test (start_time, end_time, location, description, participants) VALUES
('2024-06-10 15:00:00', '2024-06-10 17:00:00', '华南师范大学', '与导师讨论毕业论文', '张三, 李四');"""

sql2 = """INSERT INTO test (start_time, end_time, location, description, participants) VALUES
('2024-06-11 09:30:00', '2024-06-11 12:00:00', '深圳会展中心', '参加技术研讨会', '王五');"""

sql3 = """INSERT INTO test (start_time, end_time, location, description, participants) VALUES
('2024-06-12 14:00:00', '2024-06-12 17:00:00', NULL, '图书馆自习', NULL);"""

sql4 = """INSERT INTO test (start_time, end_time, location, description, participants) VALUES
('2024-06-13 10:00:00', '2024-06-13 12:00:00', '广州体育馆', '篮球比赛', '赵六, 孙七, 周八');"""

c.execute(sql1)
c.execute(sql2)


conn.commit()
conn.close()