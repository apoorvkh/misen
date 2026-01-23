---
icon: lucide/rocket
---

# Project Setup

To get started, we expect your research project to be structured as a Python [**package**](https://packaging.python.org). Among other things, this allows your project to be `import`-able and `pip` installable, so that it can be used by other projects downstream!

The best way to initialize a new project is to [install uv](https://docs.astral.sh/uv/#installation) and:

```bash
uv init my-project --package --python 3.13
cd my-project
uv sync
uv add numpy  # or other dependencies
```

This will create the following file structure. You can put your code in `src/my_project`!

```
my-project
├── pyproject.toml
├── README.md
├── src
│   └── my_project
│       └── __init__.py
└── uv.lock
```

You can then `import my_project`. And you should run your code as *modules*: for example `uv run -m src.my_project.__init__`, rather than `python src/my_project/__init__.py`.

Note about end-users. Note about system packages.
