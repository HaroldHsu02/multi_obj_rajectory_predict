import csv

def extract_csv_rows_columns(input_file, output_file, rows, cols):
    """
    截取CSV文件的前m行和前n列，并将结果保存到新的CSV文件中。

    参数:
        input_file (str): 输入CSV文件的路径。
        output_file (str): 输出CSV文件的路径。
        m (int): 需要截取的行数。
        n (int): 需要截取的列数。
    """
    try:
        with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
                open(output_file, mode='w', newline='', encoding='utf-8') as outfile:

            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            # 遍历输入文件的行
            for i, row in enumerate(reader):
                # 如果当前行数小于m，则处理该行
                if i < rows:
                    # 截取前n列
                    truncated_row = row[:cols]
                    # 将截取后的行写入输出文件
                    writer.writerow(truncated_row)
                else:
                    # 如果已经处理了m行，则停止
                    break

        print(f"成功截取前{rows}行和前{cols}列，结果已保存到 {output_file}")
    except FileNotFoundError:
        print(f"错误：文件 {input_file} 未找到。")
    except Exception as e:
        print(f"发生错误：{e}")

# 示例用法
input_csv = 'rome_traces_coordinate.csv'  # 输入文件名
output_csv = 'Cut_rome_traces_coordinate.csv'  # 输出文件名
m = 30  # 截取的行数
n = 10  # 截取的列数

extract_csv_rows_columns(input_csv, output_csv, m, n)