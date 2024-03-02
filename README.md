# embedding-workflow
Generate language model embeddings.

## Installation

To install the package, run the following command:
```bash
git clone git@github.com:ramanathanlab/embedding-workflow.git
cd embedding-workflow
pip install -e .
```

## Usage
Then to create embeddings, run the following command:
```bash
nohup python -m embedding_workflow.distributed_inference --config examples/your-config.yaml &> nohup.out &
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
