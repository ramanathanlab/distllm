# Local vLLM Server Booting Feature

This feature allows you to automatically boot up your own vLLM servers instead of connecting to existing ones.

## New Configuration Fields

### VLLMGeneratorSettings

The following new fields have been added to support local vLLM server booting:

- **`boot_local`** (bool, default: False): Whether to boot a local vLLM server
- **`hf_model_id`** (str, required if boot_local=True): Huggingface model ID to load
- **`auto_port`** (bool, default: True): Automatically find available port when booting locally
- **`local_host`** (str, default: "127.0.0.1"): Host to bind local vLLM server to
- **`vllm_args`** (dict, optional): Additional arguments for vLLM server
- **`server_startup_timeout`** (int, default: 120): Timeout in seconds to wait for server startup

## Usage Examples

### 1. YAML Configuration (Recommended)

```yaml
model:
  generator:
    generator_type: "vllm"
  
  generator_settings:
    # Traditional fields (still required)
    server: "localhost"
    port: 8000
    api_key: "CELS"
    model: "meta-llama/Llama-2-7b-chat-hf"
    temperature: 0.0
    max_tokens: 1024
    
    # New local vLLM server booting fields
    boot_local: true
    hf_model_id: "meta-llama/Llama-2-7b-chat-hf"
    auto_port: true
    local_host: "127.0.0.1"
    server_startup_timeout: 120
    
    # Optional: Additional vLLM server arguments
    vllm_args:
      tensor_parallel_size: 1
      max_model_len: 4096
      trust_remote_code: false
      gpu_memory_utilization: 0.9
      # Chat template for models that don't have one (e.g., OLMo-7B-hf)
      # chat_template: "/path/to/chat_template.jinja"
```

### 2. Command Line Usage

```bash
python rag_argonium_score_parallel_v2.py --config sample_local_vllm_config.yaml
```

## How It Works

1. **Server Startup**: When `boot_local=True`, the VLLMGenerator will:
   - Find an available port (if `auto_port=True`)
   - Start a local vLLM server with the specified `hf_model_id`
   - Wait for the server to be ready (up to `server_startup_timeout` seconds)

2. **Processing**: Once the server is running, it operates exactly like a remote vLLM server

## Chat Template Fix for Models Without Default Templates

Some models (like `allenai/OLMo-7B-hf`) don't include a default chat template. With transformers v4.44+, vLLM requires explicit chat templates for such models.

### Quick Fix

Add a `chat_template` parameter to your `vllm_args`:

```yaml
vllm_args:
  # ... other args ...
  chat_template: "/path/to/your/chat_template.jinja"
```

### Example Chat Templates

1. **Simple template** (`olmo_chat_template.jinja`):
```jinja
{% for message in messages %}
{%- if message['role'] == 'user' -%}
User: {{ message['content'] }}

{%- elif message['role'] == 'assistant' -%}
Assistant: {{ message['content'] }}

{%- elif message['role'] == 'system' -%}
{{ message['content'] }}

{%- endif -%}
{%- endfor -%}
Assistant:
```

2. **Inline template** (directly in YAML):
```yaml
vllm_args:
  chat_template: "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}Assistant:"
```

### Complete OLMo-7B-hf Example

```yaml
model:
  generator:
    generator_type: "vllm"
  generator_settings:
    boot_local: true
    hf_model_id: "allenai/OLMo-7B-hf"
    vllm_args:
      tensor_parallel_size: 8
      max_model_len: 2048
      chat_template: "/home/user/projects/distllm/olmo_chat_template.jinja"
```

3. **Cleanup**: The server is automatically shut down when:
   - The script completes normally
   - An error occurs (via try/finally blocks)
   - The script is interrupted (SIGINT/SIGTERM signals)

## Available vLLM Arguments

You can pass additional arguments to the vLLM server via the `vllm_args` field:

```yaml
vllm_args:
  tensor_parallel_size: 2           # Use 2 GPUs
  max_model_len: 8192              # Maximum sequence length
  trust_remote_code: true          # Trust remote code for custom models
  gpu_memory_utilization: 0.8      # Use 80% of GPU memory
  dtype: "float16"                 # Use float16 precision
  quantization: "awq"              # Use AWQ quantization
  seed: 42                         # Random seed for reproducibility
```

## Benefits

1. **Automatic Management**: No need to manually start/stop vLLM servers
2. **Port Management**: Automatically finds available ports to avoid conflicts
3. **Proper Cleanup**: Ensures servers are shut down properly
4. **Flexibility**: Easy switching between local and remote servers
5. **Configuration**: All settings in one YAML file

## Requirements

- vLLM must be installed (`pip install vllm`)
- Sufficient GPU memory for the model
- Network ports available for the server

## Error Handling

The system includes robust error handling:
- Server startup timeout detection
- Automatic cleanup on errors
- Signal handling for graceful shutdown
- Detailed error messages for troubleshooting

## Migration from Existing Setup

To migrate from using external vLLM servers to local booting:

1. Add `boot_local: true` to your configuration
2. Set `hf_model_id` to your desired Huggingface model
3. Optionally configure `vllm_args` for your specific needs
4. The existing `server`, `port`, and `model` fields are still required but may be overridden

## Troubleshooting

**Server won't start**: Check GPU memory availability and model compatibility
**Port conflicts**: Enable `auto_port: true` or manually specify a different port
**Slow startup**: Increase `server_startup_timeout` for large models
**Memory issues**: Adjust `gpu_memory_utilization` in `vllm_args` 