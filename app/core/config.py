from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Sklearn Docs RAG"
    app_version: str = "0.1.0"
    log_level: str = "INFO"
    debug: bool = False


settings = Settings()
