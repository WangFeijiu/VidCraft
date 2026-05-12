"""Image-to-Video pipelines — AI narration analysis + audio preview + video generation."""
import base64
import io
import json
import re
import shutil
import subprocess
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

from app.config import Settings
from app.services.llm import call_llm, call_llm_vision
from app.utils.paths import project_dir
from app.utils.state import i2v_save, i2v_state

NARRATION_STYLES = {
    "documentary": "请用纪录片旁白的风格，声音平和、娓娓道来，像在讲一个故事。",
    "humor": "请用幽默风趣的风格，语言活泼有趣，适当加入轻松的比喻和调侃。",
    "story": "请用讲故事的风格，语言生动有画面感，像一个说书人在娓娓道来。",
    "educational": "请用科普解说的风格，语言准确严谨但通俗易懂，像在讲解一个知识。",
    "product": "请用产品介绍的风格，语言专业有说服力，突出亮点和特点。",
    "news": "请用新闻报道的风格，语言正式客观，像新闻播报员在报道。",
}


def _i2v_dir(name: str, settings: Settings) -> Path:
    return settings.img2vid_dir / name


def pipeline_i2v_analyze(name: str, theme: str, style: str, settings: Settings) -> None:
    d = _i2v_dir(name, settings)
    try:
        import PIL.Image
        images_dir = d / "images"
        img_files = sorted(images_dir.glob("img_*"))
        n = len(img_files)
        if n == 0:
            raise ValueError("没有图片")

        image_data = []
        for i, f in enumerate(img_files):
            i2v_save(name, settings, stage="analyzing",
                     msg=f"正在读取图片 {i + 1}/{n}...",
                     generate_progress=[0, n])
            img = PIL.Image.open(f)
            img.thumbnail((1024, 1024))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            image_data.append({"idx": i, "b64": img_b64})

        style_hint = NARRATION_STYLES.get(style, "")
        style_line = f"\n旁白风格：{style_hint}" if style_hint else ""

        analyses = [None] * n
        done_count = {"v": 0}
        done_lock = threading.Lock()

        i2v_save(name, settings, stage="analyzing",
                 msg=f"正在并行分析 {n} 张图片...",
                 generate_progress=[0, n])

        def _analyze_one(i: int, b64: str):
            single_prompt = (
                f"请详细分析这张图片（第 {i + 1}/{n} 张）。"
                "列出你看到的所有文字内容、图表数据、核心概念或场景描述。"
                "请尽可能详细和准确，不要编造。"
            )
            analysis_text = ""
            for attempt in range(2):
                try:
                    resp = call_llm_vision(
                        settings, single_prompt, [b64],
                        system="你是一个图片分析专家。请详细准确地描述图片内容，特别要提取所有可见文字。",
                        timeout=45,
                    )
                    analysis_text = resp.strip()
                    break
                except Exception as e:
                    logger.warning(f"[i2v] image {i + 1} attempt {attempt + 1} failed: {e}")
                    if attempt == 1:
                        analysis_text = "（此图片未能成功识别，建议手动编辑旁白）"
            return i, analysis_text

        max_workers = min(6, n)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_analyze_one, item["idx"], item["b64"]) for item in image_data]
            for fut in as_completed(futures):
                i, analysis_text = fut.result()
                analyses[i] = {
                    "image_idx": i,
                    "analysis": analysis_text,
                    "skipped": not analysis_text or "未能成功识别" in analysis_text,
                }
                with done_lock:
                    done_count["v"] += 1
                    progress = done_count["v"]
                i2v_save(name, settings, stage="analyzing",
                         msg=f"已完成 {progress}/{n} 张图片分析...",
                         generate_progress=[progress, n])

        i2v_save(name, settings, stage="analyzing", msg="正在生成旁白...",
                 generate_progress=[n, n])

        analyses_text = "\n\n".join(
            f"【图片 {a['image_idx'] + 1} 分析】\n{a['analysis']}" for a in analyses
        )
        narration_prompt = (
            f"你是一个视频旁白生成专家。以下是 {n} 张图片的详细分析：\n\n"
            f"{analyses_text}\n\n"
            f"请基于以上分析，为每张图片生成 2-3 句旁白。要求：\n"
            f"- 旁白必须基于分析中的实际内容，禁止编造\n"
            f"- 口语化，适合朗读配音\n"
            f"- 多张图片之间逻辑连贯、过渡自然\n"
            f"{style_line}\n\n"
            f"输出 JSON 数组，每个元素："
            f'{{"image_idx": 序号(从0开始), "narration": "旁白文字"}}'
        )
        resp = call_llm(
            settings, narration_prompt,
            system="你是一个JSON输出专家，只输出纯JSON数组，不要markdown代码块，不要任何解释文字。",
        )
        cleaned = re.sub(r"```json\s*|\s*```", "", resp).strip()
        narration = json.loads(cleaned)

        for item in narration:
            idx = item.get("image_idx", 0)
            if 0 <= idx < len(analyses):
                item["analysis"] = analyses[idx]["analysis"]

        (d / "narration.json").write_text(
            json.dumps(narration, ensure_ascii=False, indent=2), "utf-8"
        )
        i2v_save(name, settings, stage="narration_ready", msg="旁白生成完成，请查看并编辑")
    except Exception as e:
        import PIL.Image
        images_dir = d / "images"
        img_files = sorted(images_dir.glob("img_*"))
        fallback = [{"image_idx": i, "narration": ""} for i in range(len(img_files))]
        (d / "narration.json").write_text(
            json.dumps(fallback, ensure_ascii=False, indent=2), "utf-8"
        )
        i2v_save(name, settings, stage="narration_ready",
                 msg=f"AI 分析失败（{type(e).__name__}），已生成空旁白模板，请手动填写。")
        logger.error(f"[i2v] analyze fallback: {e}\n{traceback.format_exc()}")


def pipeline_i2v_preview_audio(name: str, settings: Settings) -> None:
    d = _i2v_dir(name, settings)
    try:
        import numpy as np
        import soundfile as sf
        import torchaudio  # noqa: F401

        from app.dependencies import get_cosyvoice_model
        model, model_name = get_cosyvoice_model()

        narration = json.loads((d / "narration.json").read_text("utf-8"))
        n_items = len(narration)
        if not n_items:
            raise ValueError("没有旁白内容")

        i2v_save(name, settings, stage="audio_preview", msg="加载语音模型...",
                 generate_progress=[0, n_items])

        voice_sample = d / "voice_sample.wav"
        if not voice_sample.exists():
            voice_sample = Path(settings.COSYVOICE_DIR) / "asset" / "zero_shot_prompt.wav"

        prompt_text = "大家好，欢迎来到我的频道。今天我要和大家分享一个非常有趣的话题。"
        if "CosyVoice3" in (model_name or ""):
            full_prompt = "You are a helpful assistant.<|endofprompt|>" + prompt_text
        else:
            full_prompt = prompt_text

        SR = model.sample_rate
        rec_dir = d / "recordings"
        rec_dir.mkdir(exist_ok=True)

        for i, item in enumerate(narration):
            text = item.get("narration", "").strip()
            i2v_save(name, settings, stage="audio_preview",
                     msg=f"生成配音 {i + 1}/{n_items}...",
                     generate_progress=[i + 1, n_items])
            out_wav = rec_dir / f"s_{i + 1:03d}.wav"
            if len(text) < 5:
                silence = np.zeros(int(SR * 0.5), dtype=np.float32)
                sf.write(str(out_wav), silence, SR)
                continue
            try:
                wav_list = []
                for j, result in enumerate(model.inference_zero_shot(
                        text, full_prompt, str(voice_sample), stream=False)):
                    wav = result["tts_speech"].cpu().numpy().squeeze()
                    wav_list.append(wav)
                if wav_list:
                    combined = np.concatenate(wav_list)
                    sf.write(str(out_wav), combined, SR)
            except RuntimeError as e:
                logger.warning(f"[i2v] TTS failed for segment {i + 1}: {e}")
                silence = np.zeros(int(SR * 0.5), dtype=np.float32)
                sf.write(str(out_wav), silence, SR)

        i2v_save(name, settings, stage="audio_preview", msg="配音生成完成，请试听",
                 generate_progress=[n_items, n_items])
    except Exception as e:
        i2v_save(name, settings, stage="error",
                 msg=f"配音生成失败: {e}\n{traceback.format_exc()}")
