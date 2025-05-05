import os
import yaml
from dotenv import load_dotenv


load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SPACE_ID = os.getenv("SPACE_ID")

with open("system_prompt.yaml", "r") as f:
    SYSTEM_PROMPT = yaml.safe_load(f)
    SYSTEM_PROMPT = SYSTEM_PROMPT["system_prompt"]