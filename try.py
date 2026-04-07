import os
path = "outputs/causal_graph/"
if os.path.exists(path):
    print(os.listdir(path))
else:
    print(f"{path} does not exist")