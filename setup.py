from setuptools import setup, find_packages

setup(
    name="MyRL",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # 列出你的项目依赖的其他包
        # 例如: "numpy>=1.20", "torch>=1.10"
        # 如果已经在cleanRL_CA中安装，可以省略
        # 如果之后需要添加新的依赖项，只需更新setup.py中的install_requires部分，然后重新运行pip install -e .即可。
    ],
    python_requires=">=3.9",  # 根据你的环境设置Python版本
)