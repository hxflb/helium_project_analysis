import libcst as cst
import libcst.matchers as m
from collections import defaultdict
import os

def analyze_imports(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        source_code = file.read()
    tree = cst.parse_module(source_code)
    import_info = defaultdict(list)

    for import_stmt in tree.body:
      if isinstance(import_stmt, cst.Import):
        for imp_alias in import_stmt.names:
          import_info['module'].append(imp_alias.name.value)
          if imp_alias.asname:
            import_info['alias'].append(f"{imp_alias.name.value} as {imp_alias.asname.name.value}")
          else:
              import_info['alias'].append(imp_alias.name.value)


      if isinstance(import_stmt, cst.ImportFrom):
          for import_alias in import_stmt.names:
             import_info['from_module'].append(f"from {import_stmt.module.value} import {import_alias.name.value}")
             if import_alias.asname:
              import_info['alias'].append(f"{import_alias.name.value} as {import_alias.asname.name.value}")
             else:
                import_info['alias'].append(import_alias.name.value)

    return import_info

def analyze_functions(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        source_code = file.read()
    tree = cst.parse_module(source_code)

    function_info = []
    for node in tree.body:
      if isinstance(node, cst.FunctionDef):
        func_name = node.name.value
        params = [param.name.value for param in node.params.params if isinstance(param,cst.Param)]
        has_docstring = bool(node.body.body[0] if isinstance(node.body,cst.IndentedBlock) and isinstance(node.body.body[0], cst.SimpleStatementLine) and isinstance(node.body.body[0].body[0],cst.Expr) and  isinstance(node.body.body[0].body[0].value, cst.SimpleString) else False)

        function_info.append({
            'name': func_name,
            'parameters': params,
            'has_docstring': has_docstring
          })
    return function_info


def analyze_function_calls(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        source_code = file.read()
    tree = cst.parse_module(source_code)
    call_info = []
    for node in tree.body:
        if isinstance(node,cst.FunctionDef):
            for call_node in m.findall(node,m.Call()):
                call_data = {}
                if isinstance(call_node.func,cst.Name):
                    call_data['name'] = call_node.func.value
                elif isinstance(call_node.func,cst.Attribute):
                  call_data['name'] = call_node.func.attr.value
                  call_data['from_obj'] = call_node.func.value.value if isinstance(call_node.func.value, cst.Name) else  str(call_node.func.value)
                else:
                    call_data['name'] = str(call_node.func)
                call_data['args'] = [get_value(arg.value) for arg in call_node.args]
                call_info.append(call_data)
    return call_info

def analyze_comments(file_path):
     with open(file_path, 'r', encoding='utf-8') as file:
        source_code = file.read()
     tree = cst.parse_module(source_code)
     comment_count = len(list(m.findall(tree, m.Comment())))
     return comment_count


def analyze_variables(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        source_code = file.read()
    tree = cst.parse_module(source_code)

    variable_info = defaultdict(list)
    for node in tree.body:
      if isinstance(node,cst.FunctionDef):
        for assign_node in m.findall(node, m.Assign()):
             for target in assign_node.targets:
                if isinstance(target.target,cst.Name):
                  var_name = target.target.value
                  value = get_value(assign_node.value)
                  variable_info[var_name].append({
                    'assigned_in': node.name.value,
                    'value': value
                  })
    return variable_info

def get_call_info(call_node):
    call_info = {}
    if isinstance(call_node.func, cst.Name):
        call_info['name'] = call_node.func.value
    elif isinstance(call_node.func, cst.Attribute):
       call_info['name'] = call_node.func.attr.value
       call_info['from_obj'] = call_node.func.value.value if isinstance(call_node.func.value, cst.Name) else str(call_node.func.value)
    else:
         call_info['name'] = str(call_node.func)
    call_info['args'] = [get_value(arg) for arg in call_node.args]
    return call_info

def get_attribute_info(attribute_node):
  return {
      'name': attribute_node.attr.value,
      'from_obj': attribute_node.value.value if isinstance(attribute_node.value, cst.Name) else str(attribute_node.value)
  }
def get_binary_op_info(binary_op_node):
    return {
        'left': get_value(binary_op_node.left),
        'operator': str(binary_op_node.operator),
        'right': get_value(binary_op_node.right)
    }

def get_value(node):
    if isinstance(node, cst.Name):
      return node.value
    elif isinstance(node,cst.Integer):
      return node.value
    elif isinstance(node,cst.SimpleString):
      return node.value
    elif isinstance(node, cst.List):
        return [get_value(i) for i in node.elements]
    elif isinstance(node, cst.Call):
        return get_call_info(node)
    elif isinstance(node, cst.Attribute):
        return get_attribute_info(node)
    elif isinstance(node, cst.BinaryOperation):
      return get_binary_op_info(node)
    else:
        return str(node)


def analyze_control_flow(file_path):
  with open(file_path, 'r', encoding='utf-8') as file:
        source_code = file.read()
  tree = cst.parse_module(source_code)
  control_flow_info = []
  for node in tree.body:
      if isinstance(node,cst.If):
        if_info = {}
        if_info['type'] = 'if'
        if_info['test'] = str(node.test)
        if_info['body'] = get_block_statements(node.body)
        if node.orelse:
          if_info['orelse'] =  get_block_statements(node.orelse)
        control_flow_info.append(if_info)
      elif isinstance(node,cst.For):
          for_info = {}
          for_info['type'] = 'for'
          for_info['target'] = str(node.target)
          for_info['iter'] = str(node.iter)
          for_info['body'] = get_block_statements(node.body)
          control_flow_info.append(for_info)
      elif isinstance(node,cst.While):
          while_info = {}
          while_info['type'] = 'while'
          while_info['test'] = str(node.test)
          while_info['body'] = get_block_statements(node.body)
          control_flow_info.append(while_info)
  return control_flow_info


def get_block_statements(block):
    statements = []
    if isinstance(block, cst.IndentedBlock):
        for stmt in block.body:
            statements.append(str(stmt))
    elif isinstance(block, cst.SimpleStatementLine):
        for stmt in block.body:
            statements.append(str(stmt))

    return statements

def analyze_classes(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        source_code = file.read()
    tree = cst.parse_module(source_code)
    class_info = []

    for node in tree.body:
        if isinstance(node, cst.ClassDef):
            class_data = {
                'name': node.name.value,
                'methods': [],
                'attributes': []
            }

            for body_node in node.body.body:
                if isinstance(body_node, cst.FunctionDef):
                    class_data['methods'].append(body_node.name.value)
                elif isinstance(body_node, cst.Assign):
                   for target in body_node.targets:
                    if isinstance(target.target, cst.Name):
                      class_data['attributes'].append(target.target.value)
            class_info.append(class_data)

    return class_info


def analyze_file(file_path, output_file):
    print(f"Analyzing: {file_path}")
    output_file.write(f"Analyzing: {file_path}\n")

    import_info = analyze_imports(file_path)
    output_file.write("Import Statements:\n")
    output_file.write(f"Modules: {import_info['module']}\n")
    output_file.write(f"Aliases: {import_info['alias']}\n")
    output_file.write(f"From Modules: {import_info['from_module']}\n")

    function_info = analyze_functions(file_path)
    output_file.write("\nFunction Definitions:\n")
    for func in function_info:
        output_file.write(f"  - Name: {func['name']}, Parameters: {func['parameters']}, Has Docstring: {func['has_docstring']}\n")

    call_info = analyze_function_calls(file_path)
    output_file.write("\nFunction Calls:\n")
    for call in call_info:
      output_file.write(f"  - Name: {call['name']}, Args: {call.get('args',[])}, From Object: {call.get('from_obj','')}\n")

    comment_count = analyze_comments(file_path)
    output_file.write(f"\nComment Count: {comment_count}\n")

    variable_info = analyze_variables(file_path)
    output_file.write("\nVariables:\n")
    for var, values in variable_info.items():
        output_file.write(f"  - Name: {var}\n")
        for value in values:
            output_file.write(f"    - Assigned In: {value.get('assigned_in', '')}, Value: {value.get('value', '')}\n")

    control_flow_info = analyze_control_flow(file_path)
    output_file.write("\nControl Flow:\n")
    for cf in control_flow_info:
        output_file.write(f" - Type: {cf.get('type', '')}\n")
        if cf.get('type') == 'if':
            output_file.write(f"    Test: {cf.get('test', '')}\n")
            output_file.write(f"    Body: {cf.get('body', '')}\n")
            if 'orelse' in cf:
                output_file.write(f"    Orelse: {cf.get('orelse', '')}\n")
        elif cf.get('type') == 'for':
            output_file.write(f"    Target: {cf.get('target', '')}\n")
            output_file.write(f"    Iter: {cf.get('iter', '')}\n")
            output_file.write(f"    Body: {cf.get('body', '')}\n")
        elif cf.get('type') == 'while':
            output_file.write(f"    Test: {cf.get('test', '')}\n")
            output_file.write(f"    Body: {cf.get('body', '')}\n")

    class_info = analyze_classes(file_path)
    output_file.write("\nClasses:\n")
    for class_data in class_info:
        output_file.write(f"  - Name: {class_data['name']}\n")
        output_file.write(f"    Methods: {class_data['methods']}\n")
        output_file.write(f"    Attributes: {class_data['attributes']}\n")
    output_file.write("=" * 40 + "\n")


def main():
    project_root = 'helium'
    output_file_path = 'static_analysis_output.txt'

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for root, _, files in os.walk(project_root):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    analyze_file(file_path, output_file)

    print(f"Analysis complete. Results saved to {output_file_path}")


if __name__ == '__main__':
    main()