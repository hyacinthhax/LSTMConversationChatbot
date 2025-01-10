import os
import ast


imported_modules = []

def get_imported_modules(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    tree = ast.parse(content)
    imported_modules = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.add(node.module)

    return sorted(imported_modules)


user_input = input("Would you like to use another program?(or ENTER for requirements.txt)\n Program Name:  ")

if user_input == "":
    with open('requirements.txt', 'r') as file:
        requirements = [line for line in file.read().split('\n')]
        for requirement in requirements:
            version = str(os.system(f"pip show {requirement}"))
            print(f"{requirement} Version:  {version}")

else:
    get_imported_modules(user_input)
    requirements = imported_modules
    for requirement in requirements:
            version = str(os.system(f"pip show {requirement}"))
            print(f"{requirement} Version:  {version}")
