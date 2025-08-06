"""
__init__.py 文件的作用：将一个目录标记为一个Python包（package）
这样，该目录下的Python模块（.py文件）就可以被导入。
没有这个文件，Python 会将目录视为普通文件夹，而不是可导入的包。
即使是一个空文件，也能告诉 Python："这个目录应该被视为一个 Python 包"。

通过 pip install -e . 安装后，所有子包都能被正确识别
"""