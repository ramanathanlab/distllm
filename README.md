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
