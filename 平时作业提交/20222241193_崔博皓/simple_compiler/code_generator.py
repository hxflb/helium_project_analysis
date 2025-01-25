# code_generator.py

def generate_target_code(tac):
    for instr in tac.code:
        op, arg1, arg2, result = instr
        if op == 'ASSIGN':
            print(f"MOV {result}, {arg1}")
        elif op == 'ADD':
            print(f"ADD {result}, {arg1}, {arg2}")
        elif op == 'SUB':
            print(f"SUB {result}, {arg1}, {arg2}")
        elif op == 'MUL':
            print(f"MUL {result}, {arg1}, {arg2}")
        elif op == 'DIV':
            print(f"DIV {result}, {arg1}, {arg2}")
        elif op == 'LT':
            print(f"CMP {arg1}, {arg2}")
            print(f"SET_LT {result}")
        elif op == 'GT':
            print(f"CMP {arg1}, {arg2}")
            print(f"SET_GT {result}")
        elif op == 'LE':
            print(f"CMP {arg1}, {arg2}")
            print(f"SET_LE {result}")
        elif op == 'GE':
            print(f"CMP {arg1}, {arg2}")
            print(f"SET_GE {result}")
        elif op == 'EQ':
            print(f"CMP {arg1}, {arg2}")
            print(f"SET_EQ {result}")
        elif op == 'NEQ':
            print(f"CMP {arg1}, {arg2}")
            print(f"SET_NEQ {result}")
        elif op == 'IF_FALSE':
            print(f"IF_FALSE {arg1} GOTO {result}")
        elif op == 'GOTO':
            print(f"GOTO {result}")
        elif op == 'LABEL':
            print(f"LABEL {result}")
        elif op == 'FUNCTION':
            print(f"FUNCTION {arg1}")
        elif op == 'END_FUNCTION':
            print(f"END_FUNCTION {arg1}")
        elif op == 'PARAM':
            print(f"PARAM {arg1}")
        elif op == 'CALL':
            print(f"CALL {arg1}, {arg2}, {result}")
        elif op == 'RETURN':
            if arg1 is not None:
                print(f"RETURN {arg1}")
            else:
                print("RETURN")
        else:
            print(f"; 未知操作: {op}")

if __name__ == "__main__":
    from intermediate_code import ThreeAddressCode
    from compiler_parser import parser
    from lexer import lexer
    from code_generator import generate_target_code

    data = '''
    int a;
    a = 10 + 20;
    '''
    lexer.input(data)
    ast = parser.parse(data, lexer=lexer)
    if ast is None:
        print("解析失败，AST为None。")
    else:
        tac = ThreeAddressCode()
        tac.generate(ast)
        print("中间代码:")
        tac.print_code()
        print("\n目标代码:")
        generate_target_code(tac)
