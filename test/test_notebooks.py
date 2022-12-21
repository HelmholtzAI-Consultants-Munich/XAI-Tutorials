import os
import subprocess
from pytest import mark

notebook_path = []
for dirpath, dirnames, filenames in os.walk("."):
    if dirpath != './test/notebooks':
        for filename in [f for f in filenames if f.endswith(".ipynb")]:
            notebook_path.append((os.path.abspath(os.path.join(dirpath, filename))))

print(notebook_path)

@mark.parametrize('file', notebook_path)
def test_notebooks_run_without_errors(file):
    subprocess.check_output(f"jupyter nbconvert --output-dir='./test/notebooks' --to notebook --execute {file}".split())