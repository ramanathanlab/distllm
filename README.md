# distllm
[![PyPI version](https://badge.fury.io/py/distllm.svg)](https://badge.fury.io/py/distllm)

Distributed Inference for Large Language Models.
- Create embeddings for large datasets at scale.
- Generate text using language models at scale.
- Semantic similarity search using Faiss.

## Installation

distllm is available on PyPI and can be installed using pip:
```bash
pip install distllm
```

To install the package on Polaris@ALCF as of 12/12/2024, run the following command:
```bash
git clone git@github.com:ramanathanlab/distllm.git
cd distllm
module use /soft/modulefiles; module load conda
conda create -n distllm python=3.12 -y
conda activate distllm-12-12
pip install faiss-gpu-cu12
pip install vllm
pip install -e .
python -m nltk.downloader punkt
```

### Protein Embedding Installation
For ESMC, you can install the following package:
```bash
pip install esm
```

For ESM2, you can install the following package:
```bash
pip install flash-attn --no-build-isolation
pip install faesm[flash_attn]
```
Or, if you want to forego flash attention and just use SDPA
```bash
pip install faesm
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

To run smaller datasets on a single GPU, you can use the following command:
```bash
distllm embed --encoder_name auto --pretrained_model_name_or_path pritamdeka/S-PubMedBert-MS-MARCO --data_path /lus/eagle/projects/FoundEpidem/braceal/projects/metric-rag/data/parsed_pdfs/LUCID.small.test/parsed_pdfs --data_extension jsonl --output_path cli_test_lucid --dataset_name jsonl_chunk --batch_size 512 --chunk_batch_size 512 --buffer_size 4 --pooler_name mean --embedder_name semantic_chunk --writer_name huggingface --quantization --eval_mode
```

Or using a larger model on a single GPU, such as Salesforce/SFR-Embedding-Mistral:
```bash
distllm embed --encoder_name auto --pretrained_model_name_or_path Salesforce/SFR-Embedding-Mistral --data_path /lus/eagle/projects/FoundEpidem/braceal/projects/metric-rag/data/parsed_pdfs/LUCID.small.test/parsed_pdfs --data_extension jsonl --output_path cli_test_lucid_sfr_mistral --dataset_name jsonl_chunk --batch_size 16 --chunk_batch_size 2 --buffer_size 4 --pooler_name last_token --embedder_name semantic_chunk --writer_name huggingface --quantization --eval_mode
```

To merge the HF dataset files, you can use the following command:
```bash
distllm merge --writer_name huggingface --dataset_dir /lus/eagle/projects/FoundEpidem/braceal/projects/metric-rag/data/semantic_chunks/lit_covid_part2.PubMedBERT/embeddings --output_dir lit_covid_part2.PubMedBERT.merge
```

To generate text using a language model, you can use the following command:
```bash
distllm generate --input_dir cli_test_lucid/ --output_dir cli_test_generate --top_p 0.95
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
To release a new version of distllm to PyPI:

1. Merge the develop branch into the main branch with an updated version number in pyproject.toml.
2. Make a new release on GitHub with the tag and name equal to the version number.
3. Clone a fresh distllm repository and run the installation commands above.
4. Run the following commands from the main branch:
```bash
rm -r dist
python3 -m build
twine upload dist/*
```
