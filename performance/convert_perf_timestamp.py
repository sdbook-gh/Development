import sys
import datetime

def main():
    if len(sys.argv) < 4:
        return 1
    
    # 获取命令行参数
    input_real_time_ns = int(sys.argv[1])
    
    # 将纳秒时间戳转换为datetime对象
    real_time_stamp = datetime.datetime.fromtimestamp(input_real_time_ns / 1e9)
    
    # 输出原始时间戳的日期和时间
    print(f"Date: {real_time_stamp.strftime('%Y-%m-%d')}")
    print(f"Time: {real_time_stamp.strftime('%H:%M:%S')}")
    
    # 计算时间差
    ms = int(sys.argv[2]) / 1e6
    input_ms = float(sys.argv[3]) * 1e3
    diff_ms = input_ms - ms
    
    print(f"diff_ms: {int(diff_ms)}")
    
    # 计算结果时间戳
    result_time_stamp = real_time_stamp + datetime.timedelta(milliseconds=diff_ms)
    
    print(f"ms: {int(ms)}")
    
    # 输出结果时间戳的日期和时间
    print(f"Date: {result_time_stamp.strftime('%Y-%m-%d')}")
    print(f"Time: {result_time_stamp.strftime('%H:%M:%S')}")
    
    return 0

if __name__ == "__main__":
    main()
