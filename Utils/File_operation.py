import os
import datetime


class FileOperation:
    """保存文件，filepath指的是文件路径及文件名，文件类型由filetype指定"""

    @staticmethod
    def save(data, filepath, filetype):
        with open(filepath + "." + filetype, "a", encoding="utf-8") as f:
            f.write(data)

    """获取当前文件所在目录"""

    @staticmethod
    def get_BASE_DIR():
        BASE_DIR = os.path.dirname(__file__)  # 当前文件所在目录
        # print("path:" + BASE_DIR)
        # BASE_DIR = BASE_DIR + "/"
        # print("BASE_DIR:" + BASE_DIR)

        return BASE_DIR

    """获取当前时间的字符串"""

    @staticmethod
    def get_datetime_str(style):
        cur_time = datetime.datetime.now()

        date_str = cur_time.strftime("%y%m%d")
        time_str = cur_time.strftime("%H%M%S")

        if style == "date":
            return date_str
        elif style == "time":
            return time_str
        else:
            return date_str + "_" + time_str
