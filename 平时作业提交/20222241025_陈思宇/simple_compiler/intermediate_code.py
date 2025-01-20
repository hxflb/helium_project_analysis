# intermediate_code.py

class ThreeAddressCode:
    def __init__(self):
        self.code = []
        self.temp_count = 0

    def new_temp(self):
        temp = f't{self.temp_count}'
        self.temp_count += 1
        return temp

    def add(self, operation, arg1, arg2, result):
        self.code.append((operation, arg1, arg2, result))

    def generate(self, ast):
        if ast.type == 'program':
            for child in ast.children:
                self.generate(child)
        elif ast.type == 'var_declaration':
            # 变量声明不需要中间代码
            pass
        elif ast.type == 'fun_declaration':
            fun_name = ast.children[1].leaf
            self.add('FUNCTION', fun_name, None, None)
            self.generate(ast.children[3])  # compound_stmt
            self.add('END_FUNCTION', fun_name, None, None)
        elif ast.type == 'compound_stmt':
            for decl in ast.children[0]:
                self.generate(decl)
            for stmt in ast.children[1]:
                self.generate(stmt)
        elif ast.type == 'assign':
            var_node = ast.children[0]
            if var_node.type == 'var' and len(var_node.children) > 0:
                id_node = var_node.children[0]
                if id_node.type == 'ID':
                    var_name = id_node.leaf
                else:
                    raise Exception("Invalid var node in assign")
            else:
                raise Exception("Invalid var node in assign")
            expr_result = self.generate(ast.children[1])
            self.add('ASSIGN', expr_result, None, var_name)
            return var_name
        elif ast.type in ['addop', 'mulop', 'relop']:
            left = self.generate(ast.children[0])
            right = self.generate(ast.children[1])
            temp = self.new_temp()
            op_map = {
                '+': 'ADD',
                '-': 'SUB',
                '*': 'MUL',
                '/': 'DIV',
                '<': 'LT',
                '>': 'GT',
                '<=': 'LE',
                '>=': 'GE',
                '==': 'EQ',
                '!=': 'NEQ'
            }
            operation = op_map.get(ast.leaf, ast.type.upper())
            self.add(operation, left, right, temp)
            return temp
        elif ast.type == 'number':
            temp = self.new_temp()
            self.add('ASSIGN', ast.leaf, None, temp)
            return temp
        elif ast.type == 'var':
            var_node = ast.children[0]
            var_name = var_node.leaf
            return var_name
        elif ast.type == 'if':
            condition = self.generate(ast.children[0])
            label_else = self.new_temp()
            label_end = self.new_temp()
            self.add('IF_FALSE', condition, None, label_else)
            self.generate(ast.children[1])
            self.add('GOTO', None, None, label_end)
            self.add('LABEL', None, None, label_else)
            self.add('LABEL', None, None, label_end)
        elif ast.type == 'if_else':
            condition = self.generate(ast.children[0])
            label_else = self.new_temp()
            label_end = self.new_temp()
            self.add('IF_FALSE', condition, None, label_else)
            self.generate(ast.children[1])
            self.add('GOTO', None, None, label_end)
            self.add('LABEL', None, None, label_else)
            self.generate(ast.children[2])
            self.add('LABEL', None, None, label_end)
        elif ast.type == 'while':
            label_start = self.new_temp()
            label_end = self.new_temp()
            self.add('LABEL', None, None, label_start)
            condition = self.generate(ast.children[0])
            self.add('IF_FALSE', condition, None, label_end)
            self.generate(ast.children[1])
            self.add('GOTO', None, None, label_start)
            self.add('LABEL', None, None, label_end)
        elif ast.type == 'return_stmt':
            if len(ast.children) == 1:
                expr_result = self.generate(ast.children[0])
                self.add('RETURN', expr_result, None, None)
            else:
                self.add('RETURN', None, None, None)
        elif ast.type == 'call':
            fun_name = ast.children[0].leaf
            args = ast.children[1]
            for arg in args:
                arg_result = self.generate(arg)
                self.add('PARAM', arg_result, None, None)
            temp = self.new_temp()
            self.add('CALL', fun_name, len(args), temp)
            return temp
        else:
            for child in ast.children:
                self.generate(child)
        return None

    def print_code(self):
        for instr in self.code:
            print(instr)
