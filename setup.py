from setuptools import setup, find_packages

setup(
    name="text-to-sql-finetuning",
    version="1.0.0",
    description="Fine-tune LLMs for Text-to-SQL generation using LoRA on WikiSQL",
    author="Rajat Joshi",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "trl>=0.7.0",
        "bitsandbytes>=0.41.0",
        "accelerate>=0.25.0",
        "datasets>=2.14.0",
        "pyyaml>=6.0",
        "sqlparse>=0.4.4",
    ],
    entry_points={
        "console_scripts": [
            "sql-train=scripts.train:main",
            "sql-eval=scripts.eval:main",
            "sql-infer=scripts.infer:main",
        ],
    },
)