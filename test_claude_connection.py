# # File: test_claude_connection.py
# import asyncio
# import os
# import sys
# from dotenv import load_dotenv
# import httpx
# import logging
# import json
#
# # Configure logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[logging.StreamHandler(sys.stdout)]
# )
# logger = logging.getLogger("claude-test")
#
# # Load environment variables from .env file
# load_dotenv()
#
#
# async def test_claude_connection():
#     """Test direct connection to Claude API."""
#     # Get configuration from environment variables
#     api_key = os.getenv("ANTHROPIC_API_KEY")
#     api_url = os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com/v1/messages")
#     model = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
#
#     logger.info(f"Testing Claude API connection")
#     logger.info(f"API URL: {api_url}")
#     logger.info(f"API Key: {api_key[:5]}..." if api_key else "API Key: Not found")
#     logger.info(f"Model: {model}")
#
#     if not api_key:
#         logger.error("ERROR: ANTHROPIC_API_KEY not found in environment variables or .env file")
#         return False
#
#     # Create the request payload
#     payload = {
#         "model": model,
#         "max_tokens": 100,
#         "messages": [
#             {"role": "user", "content": "Say hello world"}
#         ]
#     }
#
#     # Prepare headers
#     headers = {
#         "x-api-key": api_key,
#         "anthropic-version": "2023-06-01",
#         "content-type": "application/json"
#     }
#
#     # Make the API request
#     try:
#         async with httpx.AsyncClient(timeout=30.0) as client:
#             logger.info(f"Sending request to Claude API: {api_url}")
#             logger.info(f"Headers: {headers}")
#             logger.info(f"Payload: {payload}")
#
#             response = await client.post(
#                 api_url,
#                 headers=headers,
#                 json=payload
#             )
#
#             logger.info(f"Response status code: {response.status_code}")
#
#             if response.status_code == 200:
#                 response_data = response.json()
#                 message = response_data.get('content', [{}])[0].get('text', '')
#                 logger.info(f"Response message: {message}")
#                 logger.info("Claude API connection test SUCCESSFUL!")
#                 return True
#             else:
#                 logger.error(f"ERROR: API request failed with status {response.status_code}")
#                 logger.error(f"Response: {response.text}")
#                 return False
#
#     except Exception as e:
#         logger.error(f"ERROR: Exception occurred during API call: {str(e)}")
#         logger.error(f"Exception details: {type(e).__name__}")
#         import traceback
#         logger.error(traceback.format_exc())
#         return False
#
#
# if __name__ == "__main__":
#     asyncio.run(test_claude_connection())


# File: test_custom_openai_connection.py
import asyncio
import os
import sys
import logging
import httpx

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("openai-test")


async def test_openai_connection():
    """Test connection to custom OpenAI deployment."""
    # Your custom configuration
    api_url = "https://azcorpstgapi.abc.com.qa/openai/deploymentons?api-version=2024-06-01"
    proxy_url = "http://zs-proxc.com.qa:443"
    use_proxy = False  # Based on your "UseProxy": "false" setting

    # Get API key from environment
    api_key = "98a26ff989784c8fa8704c829"
    if not api_key:
        logger.error("ERROR: API key not found in environment variable 'EV'")
        return False

    logger.info(f"Testing OpenAI API connection")
    logger.info(f"API URL: {api_url}")
    logger.info(f"API Key: {api_key[:5]}..." if api_key else "API Key: Not found")
    logger.info(f"Using Proxy: {use_proxy}")

    # Create the request payload
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello world"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }

    # Prepare headers for Azure OpenAI
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json"
    }

    # Set up proxy if enabled
    transport = None
    if use_proxy:
        logger.info(f"Using proxy: {proxy_url}")
        transport = httpx.AsyncHTTPTransport(proxy={"http://": proxy_url, "https://": proxy_url})

    # Make the API request
    try:
        async with httpx.AsyncClient(timeout=30.0, transport=transport) as client:
            logger.info(f"Sending request to OpenAI API: {api_url}")

            response = await client.post(
                api_url,
                headers=headers,
                json=payload
            )

            logger.info(f"Response status code: {response.status_code}")

            if response.status_code == 200:
                response_data = response.json()
                message = response_data['choices'][0]['message']['content']
                logger.info(f"Response message: {message}")
                logger.info("OpenAI API connection test SUCCESSFUL!")
                return True
            else:
                logger.error(f"ERROR: API request failed with status {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False

    except Exception as e:
        logger.error(f"ERROR: Exception occurred during API call: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    asyncio.run(test_openai_connection())