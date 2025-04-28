import os
import json
import re

# CONFIGURATION
REPO_PATH = "./"  # <- CHANGE THIS to the local path of your repo
OLD_BRANCH = "main"  # <- Branch currently in the links
NEW_BRANCH = "2025-HIDA-Spring"  # <- Branch you want to replace it with

# Regular expressions
COLAB_LINK_REGEX = re.compile(
    r'(https://colab\.research\.google\.com/github/[^/]+/[^/]+)/(blob)/([^/]+)/([^\s"\']+)', re.IGNORECASE
)
GIT_CLONE_REGEX = re.compile(r"(!git clone --branch )(\w+)( )(https://github\.com/[^\s]+)", re.IGNORECASE)

# Text patterns to uncomment
CELLS_TO_UNCOMMENT = [
    [
        "# from google.colab import drive",
        "# drive.mount('/content/drive')",
        "# %cd /content/drive/MyDrive",
    ],
    [
        "# %rm -r XAI-Tutorials",
        "# !git clone --branch main https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials.git",
    ],
    [
        "# %cd XAI-Tutorials",
        "# !pip install -r requirements_xai-for-cnn.txt",
        "# !pip install -r requirements_xai-for-random-forest.txt",
        "# !pip install -r requirements_xai-for-transformer.txt",
        "# %cd xai-for-cnn",
        "# %cd xai-for-random-forest",
        "# %cd xai-for-transformer",
    ],
]


def update_collab_links_in_cell(cell_source):
    def replacer(match):
        base_url, blob, branch, filepath = match.groups()
        if branch == OLD_BRANCH:
            return f"{base_url}/{blob}/{NEW_BRANCH}/{filepath}"
        else:
            return match.group(0)

    if isinstance(cell_source, list):
        return [COLAB_LINK_REGEX.sub(replacer, line) for line in cell_source]
    elif isinstance(cell_source, str):
        return COLAB_LINK_REGEX.sub(replacer, cell_source)
    else:
        return cell_source


def update_git_clone_in_cell(cell_source):
    def replacer(match):
        prefix, branch, space, repo_url = match.groups()
        if branch == OLD_BRANCH:
            return f"{prefix}{NEW_BRANCH}{space}{repo_url}"
        else:
            return match.group(0)

    if isinstance(cell_source, list):
        return [GIT_CLONE_REGEX.sub(replacer, line) for line in cell_source]
    elif isinstance(cell_source, str):
        return GIT_CLONE_REGEX.sub(replacer, cell_source)
    else:
        return cell_source


def uncomment_specific_cells(cell_source):
    if isinstance(cell_source, list):
        new_source = []
        for line in cell_source:
            modified_line = line
            for pattern_block in CELLS_TO_UNCOMMENT:
                for commented_line in pattern_block:
                    uncommented_line = commented_line.lstrip("# ").rstrip()
                    if uncommented_line in line.strip():
                        modified_line = line.replace("# ", "", 1)
                        break
            new_source.append(modified_line)
        return new_source
    return cell_source


def update_notebook(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    if "cells" in notebook:
        for cell in notebook["cells"]:
            if "source" in cell:
                cell["source"] = update_collab_links_in_cell(cell["source"])
                cell["source"] = uncomment_specific_cells(cell["source"])
                cell["source"] = update_git_clone_in_cell(cell["source"])

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
        f.write("\n")


def main():
    for root, dirs, files in os.walk(REPO_PATH):
        for file in files:

            if file.endswith(".ipynb"):
                filepath = os.path.join(root, file)
                print(f"Updating {filepath}...")
                update_notebook(filepath)


if __name__ == "__main__":
    print("------START------")
    main()
    print("------DONE------")
