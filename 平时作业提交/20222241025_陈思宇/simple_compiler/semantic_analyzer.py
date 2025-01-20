# semantic_analyzer.py

class SymbolTable:
    def __init__(self, parent=None):
        self.symbols = {}
        self.parent = parent

    def add(self, name, type):
        if name in self.symbols:
            raise Exception(f"符号 '{name}' 已在当前作用域中声明。")
        self.symbols[name] = type
        print(f"添加符号: {name} 类型: {type}")

    def get(self, name):
        if name in self.symbols:
            print(f"找到符号: {name} 类型: {self.symbols[name]}")
            return self.symbols[name]
        elif self.parent:
            print(f"在父作用域中查找符号: {name}")
            return self.parent.get(name)
        else:
            raise Exception(f"符号 '{name}' 未声明。")

def semantic_analysis(ast, symbol_table=None):
    if symbol_table is None:
        symbol_table = SymbolTable()
        print("创建全局符号表。")

    if ast.type == 'program':
        print("开始处理 program 节点。")
        for child in ast.children:
            semantic_analysis(child, symbol_table)

    elif ast.type == 'var_declaration':
        var_type_node = ast.children[0]
        var_id_node = ast.children[1]
        var_type = var_type_node.leaf
        var_name = var_id_node.leaf
        print(f"处理变量声明: {var_name} 类型: {var_type}")
        symbol_table.add(var_name, var_type)

    elif ast.type == 'fun_declaration':
        fun_type_node = ast.children[0]
        fun_id_node = ast.children[1]
        fun_type = fun_type_node.leaf
        fun_name = fun_id_node.leaf
        params = ast.children[2]
        print(f"处理函数声明: {fun_name} 返回类型: {fun_type}")
        symbol_table.add(fun_name, fun_type)
        # 新的作用域
        local_symbol_table = SymbolTable(parent=symbol_table)
        print(f"创建函数 '{fun_name}' 的局部符号表。")
        for param in params:
            param_type_node = param.children[0]
            param_id_node = param.children[1]
            param_type = param_type_node.leaf
            param_name = param_id_node.leaf
            print(f"处理函数参数: {param_name} 类型: {param_type}")
            local_symbol_table.add(param_name, param_type)
        semantic_analysis(ast.children[3], local_symbol_table)

    elif ast.type == 'compound_stmt':
        # 新的作用域
        local_symbol_table = SymbolTable(parent=symbol_table)
        print("创建新的复合语句作用域。")
        for decl in ast.children[0]:
            semantic_analysis(decl, local_symbol_table)
        for stmt in ast.children[1]:
            semantic_analysis(stmt, local_symbol_table)

    elif ast.type == 'expression_stmt':
        print("处理表达式语句。")
        if len(ast.children) == 1:

            semantic_analysis(ast.children[0], symbol_table)


    elif ast.type == 'assign':
        var_node = ast.children[0]
        id_node = var_node.children[0]
        var_name = id_node.leaf
        print(f"处理赋值语句: {var_name} = ...")
        try:
            var_type = symbol_table.get(var_name)
        except Exception as e:
            raise Exception(f"语义错误: {e}")
        expr_type = evaluate_expression(ast.children[1], symbol_table)
        print(f"赋值语句类型检查: {var_name} 类型: {var_type} 表达式类型: {expr_type}")
        if var_type != expr_type:
            raise Exception(f"赋值给 '{var_name}' 的类型不匹配。期望 {var_type}, 但得到 {expr_type}。")

    elif ast.type in ['addop', 'mulop', 'relop']:
        # 操作符节点的类型检查已经在 evaluate_expression 中处理
        pass

    elif ast.type == 'var':
        var_node = ast.children[0]
        var_name = var_node.leaf
        print(f"处理变量使用: {var_name}")
        try:
            symbol_table.get(var_name)
        except Exception as e:
            raise Exception(f"语义错误: {e}")

    elif ast.type == 'call':
        fun_id_node = ast.children[0]
        fun_name = fun_id_node.leaf
        print(f"处理函数调用: {fun_name}")
        try:
            fun_type = symbol_table.get(fun_name)
        except Exception as e:
            raise Exception(f"语义错误: {e}")

    elif ast.type == 'return_stmt':
        if len(ast.children) == 1:
            return_type = evaluate_expression(ast.children[0], symbol_table)
            print(f"处理返回语句: 返回类型 {return_type}")
        else:
            print("处理空返回语句。")
            # 处理 void 返回

    else:
        print(f"未处理的 AST 节点类型: {ast.type}")

def evaluate_expression(expr, symbol_table):
    if expr.type == 'number':
        if isinstance(expr.leaf, int):
            print(f"表达式是整数: {expr.leaf}")
            return 'int'
        else:
            print(f"表达式是浮点数: {expr.leaf}")
            return 'float'

    elif expr.type in ['addop', 'mulop', 'relop']:
        left_type = evaluate_expression(expr.children[0], symbol_table)
        right_type = evaluate_expression(expr.children[1], symbol_table)
        print(f"操作符 '{expr.leaf}' 左类型: {left_type} 右类型: {right_type}")
        if left_type != right_type:
            raise Exception(f"表达式中的类型不匹配: {left_type} {expr.leaf} {right_type}。")
        return left_type

    elif expr.type == 'var':
        var_node = expr.children[0]
        var_name = var_node.leaf
        print(f"表达式中的变量: {var_name}")
        return symbol_table.get(var_name)

    elif expr.type == 'assign':
        var_node = expr.children[0]
        id_node = var_node.children[0]
        var_name = id_node.leaf
        print(f"处理表达式赋值: {var_name} = ...")
        try:
            var_type = symbol_table.get(var_name)
        except Exception as e:
            raise Exception(f"语义错误: {e}")
        expr_type = evaluate_expression(expr.children[1], symbol_table)
        print(f"表达式赋值类型检查: {var_name} 类型: {var_type} 表达式类型: {expr_type}")
        if var_type != expr_type:
            raise Exception(f"赋值给 '{var_name}' 的类型不匹配。期望 {var_type}, 但得到 {expr_type}。")
        return var_type

    elif expr.type == 'call':
        fun_id_node = expr.children[0]
        fun_name = fun_id_node.leaf
        print(f"处理表达式中的函数调用: {fun_name}")
        try:
            fun_type = symbol_table.get(fun_name)
        except Exception as e:
            raise Exception(f"语义错误: {e}")
        # 此处可以扩展检查函数参数类型和数量
        return fun_type

    else:
        raise Exception(f"未知的表达式类型: {expr.type}")
