import subprocess
import os
import sys

def run_coverage(test_file):
    """运行代码覆盖率分析，处理文件路径问题。"""
    try:
        # 获取当前脚本的目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 构造测试文件的绝对路径
        test_file_path = os.path.join(script_dir, test_file)
        # 使用绝对路径运行测试，避免相对路径问题
        subprocess.run([sys.executable, '-m', 'coverage', 'run', '-m', 'unittest', test_file_path], check=True)
        subprocess.run(['coverage', 'report'], check=True)
        subprocess.run(['coverage', 'html'], check=True)  # 生成 HTML 报告到 htmlcov 文件夹
        print(f"代码覆盖率分析完成。HTML 报告在 {os.path.join(script_dir, 'htmlcov')} 目录下。")

    except subprocess.CalledProcessError as e:
        print(f"测试或代码覆盖率分析失败: {e}")
        print(f"错误代码: {e.returncode}") #打印错误代码，方便调试
        print(f"错误命令: {e.cmd}") #打印错误命令，方便调试

    except FileNotFoundError:
        print("coverage 命令未找到。请确保已安装 coverage.py 并正确配置环境变量。")
    except Exception as e: #捕捉其他异常
        print(f"发生未知错误: {e}")


if __name__ == "__main__":
    # 测试文件名
    test_file = "test_web_interactions.py"
    run_coverage(test_file)