import ast

source = """
def greet(name):
    return name

class User:
    def login(self):
        pass

    def logout(self):
        pass

    class Test:
        pass
"""

tree = ast.parse(source)

for node in tree.body:
    if isinstance(node, ast.FunctionDef):
        print("Standalone:", node.name)
    elif isinstance(node, ast.ClassDef):
        print("Class:", node.name)
