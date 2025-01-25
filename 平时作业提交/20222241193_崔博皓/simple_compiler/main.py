import sys
from lexer import lexer
from compiler_parser import parser
from semantic_analyzer import semantic_analysis
from intermediate_code import ThreeAddressCode
from optimizer import constant_folding
from code_generator import generate_target_code

def main():
    if len(sys.argv) != 2:
        print("使用方法: python main.py <源文件>")
        sys.exit(1)

    source_file = sys.argv[1]
    try:
        with open(source_file, 'r') as f:
            data = f.read()
    except FileNotFoundError:
        print(f"文件 '{source_file}' 未找到。")
        sys.exit(1)

    # 词法分析和语法分析
    lexer.input(data)
    ast = parser.parse(data, lexer=lexer)
    if ast is None:
        print("由于语法错误，解析失败。")
        sys.exit(1)
    print("抽象语法树:")
    print(ast)

    # 语义分析
    try:
        semantic_analysis(ast)
        print("\n语义分析成功完成。")
    except Exception as e:
        print(f"语义错误: {e}")
        sys.exit(1)

    # 中间代码生成
    tac = ThreeAddressCode()
    tac.generate(ast)
    print("\n中间代码:")
    tac.print_code()

    # 代码优化
    constant_folding(tac)
    print("\n优化后的中间代码:")
    tac.print_code()

    # 目标代码生成
    print("\n目标代码:")
    generate_target_code(tac)

if __name__ == "__main__":
    main()