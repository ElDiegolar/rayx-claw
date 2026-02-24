from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    minimax_api_key: str = ""
    claude_model: str = "claude-opus-4-6"
    minimax_model: str = "MiniMax-M2.5"
    workspace: str = "E:\\Sophia"
    persona_name: str = "Sophia"
    host: str = "127.0.0.1"
    port: int = 8080

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
