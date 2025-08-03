import pandas as pd

# 读取 CSV 文件
csv_file = "meps.csv"  # CSV 文件路径
xlsx_file = "meps.xlsx"  # 输出的 XLSX 文件路径

try:
    # 使用 pandas 读取 CSV 文件
    df = pd.read_csv(csv_file, sep=';')  # 如果分隔符不是逗号，指定为 ';'

    # 将 DataFrame 保存为 XLSX 文件
    df.to_excel(xlsx_file, index=False, engine='openpyxl')

    print(f"成功将 {csv_file} 转换为 {xlsx_file}")
except Exception as e:
    print(f"转换失败: {e}")



