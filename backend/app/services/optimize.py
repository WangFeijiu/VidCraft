"""Text optimization service — LLM-powered subtitle polishing and matching."""
import json
import re
import traceback
from pathlib import Path

from loguru import logger

from app.config import Settings
from app.services.llm import call_llm
from app.utils.paths import project_dir
from app.utils.state import save_state


def pipeline_optimize(name: str, sentences_path: str, user_description: str,
                      settings: Settings, sio) -> None:
    try:
        sentences = json.loads(Path(sentences_path).read_text("utf-8"))
        total = len(sentences)

        _emit_progress(sio, name, 0, "正在理解整体语境...", 5)

        if user_description:
            context_summary = user_description
        else:
            all_text = " ".join(s["text"] for s in sentences[:60])
            context_prompt = (
                f"以下是一段视频语音转写的前60句内容：\n{all_text}\n\n"
                "请用2-3句话总结这段内容的主题和语境（是教程、演讲、对话、还是其他？讲的是什么？）"
            )
            context_summary = call_llm(settings, context_prompt, system="简洁总结，不超过100字。")

        _emit_progress(sio, name, 0, "语境分析完成，开始润色...", 10)

        batch_size = 30
        polished = []
        for i in range(0, total, batch_size):
            batch = sentences[i:i + batch_size]
            prompt1 = (
                f"【语境】{context_summary}\n\n"
                f"以下是该视频第 {i + 1} 到第 {i + len(batch)} 句语音转写（共{len(batch)}句）：\n"
                f"{json.dumps([s['text'] for s in batch], ensure_ascii=False)}\n\n"
                "请基于上述语境，逐句润色为适合配音的书面语：\n"
                "- 去掉口头禅和填充词（这个、那个、就是、然后、对、嗯、啊）\n"
                "- 修正错别字，让语句通顺\n"
                "- 断断续续的表达整合成完整句子\n"
                "- 保留所有有实际意义的内容\n"
                "- 保持原来的句数不变\n"
                "输出JSON数组，只包含润色后的文本字符串，顺序和数量必须与输入完全一致。"
            )
            try:
                resp1 = call_llm(settings, prompt1, system="只输出纯JSON数组，不要解释。")
                cleaned1 = re.sub(r"```json\s*|\s*```", "", resp1).strip()
                texts = json.loads(cleaned1)
                if len(texts) != len(batch):
                    texts = [s["text"] for s in batch]
            except Exception as batch_err:
                raise RuntimeError(f"第 {i + 1}-{i + len(batch)} 句润色失败: {batch_err}")
            for j, s in enumerate(batch):
                polished.append({"text": texts[j] or s["text"], "start": s["start"], "end": s["end"]})
            done = min(i + len(batch), total)
            _emit_progress(sio, name, 1, "润色中...", 10 + int((done / total) * 70))

        merged = []
        merge_batch_size = 40
        for i in range(0, len(polished), merge_batch_size):
            batch = polished[i:i + merge_batch_size]
            _emit_progress(sio, name, 2, "语义整合中...",
                           80 + int(((i + len(batch)) / len(polished)) * 15))
            prompt2 = (
                f"【语境】{context_summary}\n\n"
                f"以下是第 {i + 1} 到第 {i + len(batch)} 句润色后的字幕（共{len(batch)}句）：\n"
                f"{json.dumps(batch, ensure_ascii=False)}\n\n"
                "请根据语义进行适度的合并：\n"
                "- 只合并明显是同一句话被切断的碎片\n"
                "- 大部分句子保持不动\n"
                "- 过长的句子（超过50字）可以拆分\n"
                "- 合并后：start取第一句的start，end取最后一句的end\n"
                "- 拆分后：按比例分配时间段\n\n"
                "输出JSON数组，每个元素：\n"
                "- text: 文本\n- start: 浮点数\n- end: 浮点数\n"
                "- source: 数组，对应输入的句子序号（1-based，相对于本批）"
            )
            try:
                resp2 = call_llm(settings, prompt2, system="只输出纯JSON数组，不要解释。")
                cleaned2 = re.sub(r"```json\s*|\s*```", "", resp2).strip()
                batch_result = json.loads(cleaned2)
                if isinstance(batch_result, list):
                    merged.extend(batch_result)
                else:
                    merged.extend(batch)
            except Exception:
                merged.extend(batch)

        d = project_dir(name, settings)
        (d / "sentences_optimized.json").write_text(
            json.dumps(merged, ensure_ascii=False, indent=2), "utf-8"
        )
        _emit_progress(sio, name, "done", f"优化完成：{total} 句 → {len(merged)} 句", 100,
                       result_count=len(merged))
    except Exception as e:
        _emit_progress(sio, name, "error", str(e), 0)


def _emit_progress(sio, name, step, msg, progress, **extra):
    if sio is None:
        return
    data = {"project": name, "step": step, "msg": msg, "progress": progress, **extra}
    try:
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            sio.emit("optimize_progress", data, namespace="/")
        )
    except RuntimeError:
        pass
