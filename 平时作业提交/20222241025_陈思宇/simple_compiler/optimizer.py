# optimizer.py

def constant_folding(tac):
    constants = {}
    optimized_code = []
    for instr in tac.code:
        op, arg1, arg2, result = instr
        if op == 'ASSIGN' and isinstance(arg1, (int, float)):
            constants[result] = arg1
            optimized_code.append(instr)
        elif op in ['ADD', 'SUB', 'MUL', 'DIV', 'LT', 'GT', 'LE', 'GE', 'EQ', 'NEQ']:
            if arg1 in constants:
                arg1 = constants[arg1]
            if arg2 in constants:
                arg2 = constants[arg2]
            if isinstance(arg1, (int, float)) and isinstance(arg2, (int, float)):
                if op == 'ADD':
                    value = arg1 + arg2
                elif op == 'SUB':
                    value = arg1 - arg2
                elif op == 'MUL':
                    value = arg1 * arg2
                elif op == 'DIV':
                    value = arg1 / arg2 if arg2 != 0 else 0
                elif op == 'LT':
                    value = int(arg1 < arg2)
                elif op == 'GT':
                    value = int(arg1 > arg2)
                elif op == 'LE':
                    value = int(arg1 <= arg2)
                elif op == 'GE':
                    value = int(arg1 >= arg2)
                elif op == 'EQ':
                    value = int(arg1 == arg2)
                elif op == 'NEQ':
                    value = int(arg1 != arg2)
                constants[result] = value
                optimized_code.append(('ASSIGN', value, None, result))
            else:
                optimized_code.append((op, arg1, arg2, result))
        else:
            optimized_code.append(instr)
    tac.code = optimized_code


if __name__ == "__main__":
    from intermediate_code import ThreeAddressCode
    from compiler_parser import parser
    from lexer import lexer
    from optimizer import constant_folding

    data = '''
    int a;
    a = 10 + 20;
    '''
    lexer.input(data)
    ast = parser.parse(data, lexer=lexer)
    tac = ThreeAddressCode()
    tac.generate(ast)
    print("优化前的中间代码:")
    tac.print_code()
    constant_folding(tac)
    print("\n优化后的中间代码:")
    tac.print_code()
