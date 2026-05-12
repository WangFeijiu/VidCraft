"""Voice Studio Backend — Configuration via environment variables."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    HOST: str = "127.0.0.1"
    PORT: int = 5050
    LOG_LEVEL: str = "INFO"

    DATA_DIR: Path = Path("./data")
    FFMPEG_PATH: str = "ffmpeg"
    COSYVOICE_DIR: Path = Path("./CosyVoice")
    HF_CACHE_DIR: Path = Path("./.cache/huggingface")

    PROJECTS_DIR: Path | None = None
    IMG2VID_DIR: Path | None = None
    TOOL_WORKSPACE_DIR: Path | None = None
    VOICE_CACHE_DIR: Path | None = None
    CUSTOM_VOICES_DIR: Path | None = None

    FRONTEND_DIR: Path = Path("../voice_studio/templates")

    @property
    def projects_dir(self) -> Path:
        return self.PROJECTS_DIR or self.DATA_DIR / "projects"

    @property
    def img2vid_dir(self) -> Path:
        return self.IMG2VID_DIR or self.DATA_DIR / "img2vid"

    @property
    def tool_workspace_dir(self) -> Path:
        return self.TOOL_WORKSPACE_DIR or self.DATA_DIR / "tool_workspace"

    @property
    def voice_cache_dir(self) -> Path:
        return self.VOICE_CACHE_DIR or self.DATA_DIR / "voice_cache"

    @property
    def custom_voices_dir(self) -> Path:
        return self.CUSTOM_VOICES_DIR or self.DATA_DIR / "custom_voices"

    def ensure_dirs(self) -> None:
        for d in (
            self.projects_dir,
            self.img2vid_dir,
            self.tool_workspace_dir,
            self.voice_cache_dir,
            self.custom_voices_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
