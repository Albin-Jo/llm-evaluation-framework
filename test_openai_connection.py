# File: test_openai_connection.py
import asyncio
import os
import sys
from dotenv import load_dotenv
import httpx
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("openai-test")

# Load environment variables from .env file
load_dotenv()


async def test_openai_connection():
    """Test direct connection to OpenAI API with custom configuration."""
    # Get configuration from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    api_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    proxy_url = os.getenv("OPENAI_PROXY_URL")
    organization_id = os.getenv("OPENAI_ORGANIZATION_ID")

    logger.info(f"Testing OpenAI API connection")
    logger.info(f"API URL: {api_url}")
    logger.info(f"API Key: {api_key[:5]}..." if api_key else "API Key: Not found")
    logger.info(f"Proxy URL: {proxy_url}")
    logger.info(f"Organization ID: {organization_id}")

    if not api_key:
        logger.error("ERROR: OPENAI_API_KEY not found in environment variables or .env file")
        return False

    # Create the request payload
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello world"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }

    # Prepare headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Add organization header if provided
    if organization_id:
        headers["OpenAI-Organization"] = organization_id

    # Set up proxy if provided
    transport = None
    if proxy_url:
        logger.info(f"Using proxy: {proxy_url}")
        transport = httpx.AsyncHTTPTransport(proxy={"http://": proxy_url, "https://": proxy_url})

    # Make the API request
    try:
        async with httpx.AsyncClient(timeout=30.0, transport=transport) as client:
            logger.info(f"Sending request to OpenAI API: {api_url}")
            logger.info(f"Headers: {headers}")
            logger.info(f"Payload: {payload}")

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
        logger.error(f"Exception details: {type(e).__name__}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    asyncio.run(test_openai_connection())