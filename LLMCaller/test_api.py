  import os
  from config.loader import load_llm_setup, get_global_config
  print('Global config:', get_global_config())
  print('Credentials dir:', os.environ.get('LLMCALLER_CREDENTIALS'))
  print('Checking OpenRouter key...')
  from api.api_factory import create_api_client
  config = load_llm_setup('llama-3.3-8b')
  print('Config:', config)
  try:
      client = create_api_client(config)
      print('Client created successfully!')
  except Exception as e:
      print(f'Error creating client: {e}')
  ENDOFFILE

