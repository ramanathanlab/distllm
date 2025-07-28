#!/usr/bin/env python3
"""
Test script for local vLLM server booting functionality.
This script tests various aspects of the new feature without running a full evaluation.
"""

import json
import os
import sys
import tempfile
import time
import yaml
from pathlib import Path

# Add the distllm directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from distllm.mcqa.rag_argonium_score_parallel_v2 import (
    VLLMGeneratorConfig,
    VLLMGeneratorSettings,
    VLLMGenerator,
    find_available_port,
    wait_for_server_ready,
)


def test_port_finding():
    """Test the port finding functionality."""
    print('🔍 Testing port finding...')
    try:
        port = find_available_port(start_port=8000)
        print(f'✅ Found available port: {port}')
        return True
    except Exception as e:
        print(f'❌ Port finding failed: {e}')
        return False


def test_config_validation():
    """Test configuration validation."""
    print('\n🔧 Testing configuration validation...')

    # Test 1: Valid config with boot_local=False
    try:
        config = VLLMGeneratorSettings(
            server='localhost',
            model='test-model',
            port=8000,
            api_key='test',
            boot_local=False,
        )
        print('✅ Valid config without boot_local works')
    except Exception as e:
        print(f'❌ Valid config failed: {e}')
        return False

    # Test 2: Invalid config - boot_local=True without hf_model_id
    try:
        config = VLLMGeneratorSettings(
            server='localhost',
            model='test-model',
            port=8000,
            api_key='test',
            boot_local=True,
            # Missing hf_model_id
        )
        print('❌ Should have failed - missing hf_model_id')
        return False
    except ValueError as e:
        print('✅ Correctly rejected config missing hf_model_id')

    # Test 3: Valid config with boot_local=True
    try:
        config = VLLMGeneratorSettings(
            server='localhost',
            model='test-model',
            port=8000,
            api_key='test',
            boot_local=True,
            hf_model_id='microsoft/DialoGPT-small',
        )
        print('✅ Valid config with boot_local and hf_model_id works')
    except Exception as e:
        print(f'❌ Valid boot_local config failed: {e}')
        return False

    return True


def test_vllm_generator_creation():
    """Test VLLMGenerator creation without actually starting a server."""
    print('\n🏗️ Testing VLLMGenerator creation...')

    # Test traditional config (should work as before)
    try:
        config = VLLMGeneratorConfig(
            server='localhost',
            port=8000,
            api_key='test',
            model='test-model',
            boot_local=False,
        )
        # Don't actually create the generator since we don't have a real server
        print('✅ Traditional VLLMGeneratorConfig creation works')
    except Exception as e:
        print(f'❌ Traditional config failed: {e}')
        return False

    # Test local boot config
    try:
        config = VLLMGeneratorConfig(
            server='localhost',
            port=8000,
            api_key='test',
            model='test-model',
            boot_local=True,
            hf_model_id='microsoft/DialoGPT-small',
            auto_port=True,
            server_startup_timeout=60,
        )
        print('✅ Local boot VLLMGeneratorConfig creation works')
    except Exception as e:
        print(f'❌ Local boot config failed: {e}')
        return False

    return True


def create_test_questions():
    """Create a simple test questions file."""
    questions = [
        {'question': 'What is 2+2?\nA) 3\nB) 4\nC) 5\nD) 6', 'answer': 'B) 4'},
        {'question': 'What color is the sky?', 'answer': 'Blue'},
    ]

    # Create temporary file
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', delete=False
    ) as f:
        json.dump(questions, f, indent=2)
        return f.name


def create_test_config(questions_file, use_local_vllm=True):
    """Create a test configuration file."""
    config = {
        'questions_file': questions_file,
        'model': {
            'generator': {'generator_type': 'vllm'},
            'generator_settings': {
                'server': 'localhost',
                'port': 8000,
                'api_key': 'test',
                'model': 'microsoft/DialoGPT-small',
                'temperature': 0.0,
                'max_tokens': 50,
                'boot_local': use_local_vllm,
                'hf_model_id': 'microsoft/DialoGPT-small'
                if use_local_vllm
                else None,
                'auto_port': True if use_local_vllm else False,
                'local_host': '127.0.0.1',
                'server_startup_timeout': 60,
                'vllm_args': {
                    'max_model_len': 512,  # Small for testing
                    'tensor_parallel_size': 1,
                }
                if use_local_vllm
                else None,
            },
            'grader_shortname': 'gpt41',
            'model_config_file': 'model_servers.yaml',
        },
        'rag': {
            'enabled': False  # Disable RAG for simple testing
        },
        'processing': {'parallel_workers': 1, 'verbose': True},
    }

    # Create temporary config file
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.yaml', delete=False
    ) as f:
        yaml.dump(config, f, default_flow_style=False)
        return f.name


def test_config_file_creation():
    """Test creation of configuration files."""
    print('\n📄 Testing configuration file creation...')

    try:
        # Create test questions
        questions_file = create_test_questions()
        print(f'✅ Created test questions file: {questions_file}')

        # Create test config
        config_file = create_test_config(questions_file, use_local_vllm=True)
        print(f'✅ Created test config file: {config_file}')

        # Verify config file is valid YAML
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        print('✅ Config file is valid YAML')

        # Verify required fields are present
        assert config_data['model']['generator_settings']['boot_local'] == True
        assert (
            config_data['model']['generator_settings']['hf_model_id']
            == 'microsoft/DialoGPT-small'
        )
        print('✅ Config file has correct local vLLM settings')

        # Cleanup
        os.unlink(questions_file)
        os.unlink(config_file)
        print('✅ Cleaned up test files')

        return True
    except Exception as e:
        print(f'❌ Config file creation test failed: {e}')
        return False


def test_dry_run():
    """Test a dry run without actually starting vLLM."""
    print('\n🧪 Testing dry run (config loading only)...')

    try:
        from distllm.mcqa.rag_argonium_score_parallel_v2 import MCQAConfig

        # Create test files
        questions_file = create_test_questions()
        config_file = create_test_config(questions_file, use_local_vllm=True)

        # Load and validate config
        config = MCQAConfig.from_yaml(config_file)
        print('✅ MCQA config loaded successfully')

        # Verify settings
        assert config.model.generator_settings.boot_local == True
        assert (
            config.model.generator_settings.hf_model_id
            == 'microsoft/DialoGPT-small'
        )
        print('✅ Local vLLM settings correctly loaded')

        # Cleanup
        os.unlink(questions_file)
        os.unlink(config_file)

        return True
    except Exception as e:
        print(f'❌ Dry run test failed: {e}')
        return False


def main():
    """Run all tests."""
    print('🚀 Testing Local vLLM Server Booting Feature')
    print('=' * 50)

    tests = [
        test_port_finding,
        test_config_validation,
        test_vllm_generator_creation,
        test_config_file_creation,
        test_dry_run,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f'❌ Test {test.__name__} crashed: {e}')
            failed += 1

    print('\n' + '=' * 50)
    print(f'📊 Test Results: {passed} passed, {failed} failed')

    if failed == 0:
        print('🎉 All tests passed! The local vLLM feature is ready to use.')
        print('\n📋 Next steps:')
        print('1. Install vLLM: pip install vllm')
        print('2. Edit sample_local_vllm_config.yaml with your desired model')
        print(
            '3. Run: python rag_argonium_score_parallel_v2.py --config sample_local_vllm_config.yaml'
        )
    else:
        print('⚠️ Some tests failed. Please check the implementation.')
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
