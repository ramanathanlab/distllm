# distllm
Distributed Inference for Large Language Models.
- Create embeddings for large datasets at scale.
- Generate text using language models at scale.

## Installation

To install the package, run the following command:
```bash
git clone git@github.com:ramanathanlab/distllm.git
cd distllm
pip install -e .
```

If you'd like to use the semantic chunking option, you need to install nltk:
```bash
pip install nltk==3.8.1
```

## Usage
To create embeddings at scale, run the following command:
```bash
nohup python -m distllm.distributed_embedding --config examples/your-config.yaml &> nohup.out &
```

For LLM generation at scale, run the following command:
```bash
nohup python -m distllm.distributed_generation --config examples/your-config.yaml &> nohup.out &
```

## Contributing

For development, it is recommended to use a virtual environment. The following commands will create a virtual environment, install the package in editable mode, and install the pre-commit hooks.
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -e '.[dev,docs]'
pre-commit install
```
To test the code, run the following command:
```bash
pre-commit run --all-files
tox -e py310
```
