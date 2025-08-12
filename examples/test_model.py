#!/usr/bin/env python3
"""
Quick test script to make API calls to models defined in model_servers.yaml

Usage:
    python test_model.py <model_shortname> [test_message]

Examples:
    python test_model.py oss120
    python test_model.py oss120 "What is the capital of France?"
    python test_model.py gpt-4o "Explain quantum computing"
"""

import os
import sys
import yaml
import openai
from pathlib import Path


def load_model_config(model_shortname, config_file='distllm/mcqa/model_servers.yaml'):
    """Load model configuration from the specified configuration file."""
    
    # Try to find the config file relative to script location or as absolute path
    if os.path.isabs(config_file):
        yaml_path = config_file
    else:
        # Try relative to current directory first
        if os.path.exists(config_file):
            yaml_path = config_file
        else:
            # Try relative to script directory
            script_dir = Path(__file__).parent
            yaml_path = script_dir / config_file
            if not yaml_path.exists():
                raise FileNotFoundError(f'Model config file not found: {config_file}')
    
    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f'Model config file not found: {yaml_path}')
    except yaml.YAMLError as e:
        raise ValueError(f'Error parsing YAML file: {e}')
    
    # Find the model configuration
    model_config = None
    for server in config.get('servers', []):
        if server.get('shortname') == model_shortname:
            model_config = server
            break
    
    if model_config is None:
        available_models = [
            server.get('shortname') for server in config.get('servers', [])
        ]
        raise ValueError(
            f'Model "{model_shortname}" not found in config. Available models: {available_models}'
        )
    
    return model_config


def create_openai_client(model_config):
    """Create an OpenAI client from model configuration."""
    return openai.OpenAI(
        api_key=model_config['openai_api_key'],
        base_url=model_config['openai_api_base']
    )


def test_model(model_shortname, test_message="Hello! Can you tell me a short joke?"):
    """Test a model by sending it a simple message."""
    
    print(f"Testing model: {model_shortname}")
    print(f"Test message: {test_message}")
    print("-" * 50)
    
    try:
        # Load model configuration
        model_config = load_model_config(model_shortname)
        print(f"Model config loaded:")
        print(f"  Server: {model_config['server']}")
        print(f"  API Base: {model_config['openai_api_base']}")
        print(f"  Model: {model_config['openai_model']}")
        print()
        
        # Create OpenAI client
        client = create_openai_client(model_config)
        
        # Make the API call
        print("Sending request...")
        response = client.chat.completions.create(
            model=model_config['openai_model'],
            messages=[
                {"role": "user", "content": test_message}
            ],
            temperature=0.7,
            max_tokens=512
        )
        
        # Print the response
        print("Response received!")
        print("=" * 50)
        print(response.choices[0].message.content)
        print("=" * 50)
        print(f"Usage: {response.usage}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model {model_shortname}: {e}")
        return False


def list_available_models():
    """List all available models from the config file."""
    try:
        config_file = 'distllm/mcqa/model_servers.yaml'
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        
        print("Available models:")
        print("-" * 30)
        for server in config.get('servers', []):
            shortname = server.get('shortname')
            server_name = server.get('server')
            model_name = server.get('openai_model')
            print(f"  {shortname:<15} | {server_name:<10} | {model_name}")
        
    except Exception as e:
        print(f"Error loading model list: {e}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n")
        list_available_models()
        return
    
    model_shortname = sys.argv[1]
    
    # Check for special commands
    if model_shortname in ['--list', '-l', 'list']:
        list_available_models()
        return
    
    # Get custom test message if provided
    test_message = sys.argv[2] if len(sys.argv) > 2 else "Hello! Can you tell me a short joke?"
    
    # Test the model
    success = test_model(model_shortname, test_message)
    
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
