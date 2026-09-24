x = "scans the provided text for possible secrets such as API keys, passwords, tokens, AWS access keys"

print("STRING:", repr(x))

import ast

source = "x = " + repr(x)
tree = ast.parse(source)

print("AST:", repr(tree.body[0].value.value))
