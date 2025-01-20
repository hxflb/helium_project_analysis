# lexer.py
import ply.lex as lex

# 关键词
reserved = {
    'int': 'INT',
    'float': 'FLOAT',
    'if': 'IF',
    'else': 'ELSE',
    'while': 'WHILE',
    'return': 'RETURN',
    'void': 'VOID',
}

# 定义tokens
tokens = [
    'ID',
    'NUMBER',
    'PLUS',
    'MINUS',
    'MULTIPLY',
    'DIVIDE',
    'ASSIGN',
    'LPAREN',
    'RPAREN',
    'LBRACE',
    'RBRACE',
    'COMMA',
    'SEMICOLON',
    'LT',
    'GT',
    'LE',
    'GE',
    'EQ',
    'NEQ',
] + list(reserved.values())

# 正则表达式规则
t_PLUS       = r'\+'
t_MINUS      = r'-'
t_MULTIPLY   = r'\*'
t_DIVIDE     = r'/'
t_ASSIGN     = r'='
t_LPAREN     = r'\('
t_RPAREN     = r'\)'
t_LBRACE     = r'\{'
t_RBRACE     = r'\}'
t_COMMA      = r','
t_SEMICOLON  = r';'
t_LE         = r'<='
t_GE         = r'>='
t_EQ         = r'=='
t_NEQ        = r'!='
t_LT         = r'<'
t_GT         = r'>'

# 忽略空格和制表符
t_ignore = ' \t'

# 处理标识符和关键词
def t_ID(t):
    r'[A-Za-z_][A-Za-z0-9_]*'
    t.type = reserved.get(t.value, 'ID')  # 检查是否为关键词
    return t

# 处理数字（整数和浮点数）
def t_NUMBER(t):
    r'\d+(\.\d+)?'
    if '.' in t.value:
        t.value = float(t.value)
    else:
        t.value = int(t.value)
    return t
# 处理换行
def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

# 处理注释（单行注释）
def t_comment(t):
    r'//.*'
    pass  # 忽略注释

# 处理错误
def t_error(t):
    print(f"非法字符 '{t.value[0]}' 在第 {t.lineno} 行")
    t.lexer.skip(1)

# 构建lexer
lexer = lex.lex()

# 测试lexer
if __name__ == "__main__":
    data = '''
    int a;
    float b;
    a = 10;
    b = 20.5;
    if (a < b) {
        a = a + 1;
    } else {
        a = a - 1;
    }
    while (a < 15) {
        a = a + 1;
    }
    int add(int x, int y) {
        return x + y;
    }
    int result;
    result = add(a, 5);
    '''
    lexer.input(data)
    for tok in lexer:
        print(tok)
