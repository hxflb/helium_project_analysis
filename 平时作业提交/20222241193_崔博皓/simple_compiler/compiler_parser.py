# compiler_parser.py
import ply.yacc as yacc
from lexer import tokens

# 定义AST节点
class ASTNode:
    def __init__(self, type, children=None, leaf=None):
        self.type = type
        self.children = children if children else []
        self.leaf = leaf

    def __repr__(self, level=0):
        ret = "  " * level + f"{self.type}: {self.leaf}\n"
        for child in self.children:
            if isinstance(child, ASTNode):
                ret += child.__repr__(level + 1)
            else:
                ret += "  " * (level + 1) + str(child) + "\n"
        return ret

# 操作符优先级和结合性
precedence = (
    ('left', 'EQ', 'NEQ'),
    ('left', 'LT', 'LE', 'GT', 'GE'),
    ('left', 'PLUS', 'MINUS'),
    ('left', 'MULTIPLY', 'DIVIDE'),
)

# 语法规则
def p_program(p):
    '''program : declaration_list statement_list'''
    print("解析 program 节点。")
    p[0] = ASTNode('program', p[1] + p[2])

def p_declaration_list(p):
    '''declaration_list : declaration_list declaration
                        | declaration'''
    if len(p) == 3:
        p[0] = p[1] + [p[2]]
    else:
        p[0] = [p[1]]

def p_declaration(p):
    '''declaration : var_declaration
                   | fun_declaration'''
    p[0] = p[1]

def p_var_declaration(p):
    '''var_declaration : type_specifier ID SEMICOLON'''
    id_node = ASTNode('ID', leaf=p[2])
    p[0] = ASTNode('var_declaration', [p[1], id_node])
    print(f"解析变量声明: {p[2]}")

def p_type_specifier(p):
    '''type_specifier : INT
                      | FLOAT
                      | VOID'''
    p[0] = ASTNode('type_specifier', leaf=p[1])
    print(f"解析类型说明符: {p[1]}")

def p_fun_declaration(p):
    '''fun_declaration : type_specifier ID LPAREN params RPAREN compound_stmt'''
    id_node = ASTNode('ID', leaf=p[2])
    p[0] = ASTNode('fun_declaration', [p[1], id_node, p[4], p[6]])
    print(f"解析函数声明: {p[2]}")

def p_params(p):
    '''params : param_list
              | VOID
              | empty'''
    if p[1] == 'void' or p[1] is None:
        p[0] = []
    else:
        p[0] = p[1]

def p_param_list(p):
    '''param_list : param_list COMMA param
                  | param'''
    if len(p) == 4:
        p[0] = p[1] + [p[3]]
    else:
        p[0] = [p[1]]

def p_param(p):
    '''param : type_specifier ID'''
    id_node = ASTNode('ID', leaf=p[2])
    p[0] = ASTNode('param', [p[1], id_node])
    print(f"解析函数参数: {p[2]}")

def p_compound_stmt(p):
    '''compound_stmt : LBRACE local_declarations statement_list RBRACE'''
    p[0] = ASTNode('compound_stmt', [p[2], p[3]])
    print("解析复合语句。")

def p_local_declarations(p):
    '''local_declarations : local_declarations var_declaration
                          | empty'''
    if len(p) == 3:
        p[0] = p[1] + [p[2]]
    else:
        p[0] = []

def p_statement_list(p):
    '''statement_list : statement_list statement
                      | empty'''
    if len(p) == 3:
        p[0] = p[1] + [p[2]]
    else:
        p[0] = []

def p_statement(p):
    '''statement : expression_stmt
                 | compound_stmt
                 | selection_stmt
                 | iteration_stmt
                 | return_stmt'''
    p[0] = p[1]

def p_expression_stmt(p):
    '''expression_stmt : expression SEMICOLON
                       | SEMICOLON'''
    if len(p) == 3:
        p[0] = ASTNode('expression_stmt', [p[1]])
    else:
        p[0] = ASTNode('expression_stmt')
    print("解析表达式语句。")

def p_selection_stmt(p):
    '''selection_stmt : IF LPAREN expression RPAREN statement ELSE statement
                      | IF LPAREN expression RPAREN statement'''
    if len(p) == 8:
        p[0] = ASTNode('if_else', [p[3], p[5], p[7]])
        print("解析 if-else 语句。")
    else:
        p[0] = ASTNode('if', [p[3], p[5]])
        print("解析 if 语句。")

def p_iteration_stmt(p):
    '''iteration_stmt : WHILE LPAREN expression RPAREN statement'''
    p[0] = ASTNode('while', [p[3], p[5]])
    print("解析 while 语句。")

def p_return_stmt(p):
    '''return_stmt : RETURN SEMICOLON
                   | RETURN expression SEMICOLON'''
    if len(p) == 3:
        p[0] = ASTNode('return_stmt')
    else:
        p[0] = ASTNode('return_stmt', [p[2]])
    print("解析返回语句。")

def p_expression(p):
    '''expression : var ASSIGN expression
                  | simple_expression'''
    if len(p) == 4:
        p[0] = ASTNode('assign', [p[1], p[3]], p[2])
        print("解析赋值表达式。")
    else:
        p[0] = p[1]

def p_var(p):
    '''var : ID'''
    id_node = ASTNode('ID', leaf=p[1])
    p[0] = ASTNode('var', [id_node])
    print(f"解析变量: {p[1]}")

def p_simple_expression(p):
    '''simple_expression : additive_expression relop additive_expression
                         | additive_expression'''
    if len(p) == 4:
        p[0] = ASTNode('relop', [p[1], p[3]], p[2])
    else:
        p[0] = p[1]

def p_relop(p):
    '''relop : LT
             | LE
             | GT
             | GE
             | EQ
             | NEQ'''
    p[0] = p[1]

def p_additive_expression(p):
    '''additive_expression : additive_expression addop term
                           | term'''
    if len(p) == 4:
        p[0] = ASTNode('addop', [p[1], p[3]], p[2])
    else:
        p[0] = p[1]

def p_addop(p):
    '''addop : PLUS
             | MINUS'''
    p[0] = p[1]

def p_term(p):
    '''term : term mulop factor
            | factor'''
    if len(p) == 4:
        p[0] = ASTNode('mulop', [p[1], p[3]], p[2])
    else:
        p[0] = p[1]

def p_mulop(p):
    '''mulop : MULTIPLY
             | DIVIDE'''
    p[0] = p[1]

def p_factor(p):
    '''factor : LPAREN expression RPAREN
              | var
              | call
              | NUMBER'''
    if len(p) == 4:
        p[0] = p[2]
    elif isinstance(p[1], ASTNode):
        p[0] = p[1]
    elif isinstance(p[1], (int, float)):
        p[0] = ASTNode('number', leaf=p[1])
        print(f"解析数字: {p[1]}")
    else:
        p[0] = p[1]

def p_call(p):
    '''call : ID LPAREN args RPAREN'''
    id_node = ASTNode('ID', leaf=p[1])
    p[0] = ASTNode('call', [id_node, p[3]])
    print(f"解析函数调用: {p[1]}")

def p_args(p):
    '''args : arg_list
            | empty'''
    if p[1] is None:
        p[0] = []
    else:
        p[0] = p[1]

def p_arg_list(p):
    '''arg_list : arg_list COMMA expression
                | expression'''
    if len(p) == 4:
        p[0] = p[1] + [p[3]]
    else:
        p[0] = [p[1]]

def p_empty(p):
    'empty :'
    p[0] = None

def p_error(p):
    if p:
        print(f"语法错误在 '{p.value}' (第 {p.lineno} 行)")
    else:
        print("语法错误在文件结尾")

# 构建parser
parser = yacc.yacc()

if __name__ == "__main__":
    from lexer import lexer
    data = '''
    int a;
    a = 10 + 20;
    if (a < 30) {
        a = a + 1;
    }
    '''
    lexer.input(data)
    result = parser.parse(data, lexer=lexer)
    print(result)
