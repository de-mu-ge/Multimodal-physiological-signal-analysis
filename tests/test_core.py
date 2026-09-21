import openpyxl

# 创建一个新的工作簿
wb = openpyxl.Workbook()

# 获取当前活动的工作表（默认名为 Sheet）
ws = wb.active
ws.title = "销售数据"

# 写入表头
ws.append(["产品名称", "数量", "单价"])

# 写入具体行数据
ws.append(["苹果", 50, 5.5])
ws.append(["香蕉", 30, 3.0])
ws.append(["橙子", 20, 4.2])

# 保存为 xlsx 文件
wb.save("sales_data.xlsx")
