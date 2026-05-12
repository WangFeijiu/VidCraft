---
stepsCompleted: ['step-01-init', 'step-02-discovery', 'step-02b-vision', 'step-02c-executive-summary', 'step-03-success', 'step-04-journeys', 'step-05-domain-skipped', 'step-06-innovation-skipped', 'step-07-project-type', 'step-08-scoping', 'step-09-functional', 'step-10-nonfunctional', 'step-11-polish', 'step-12-complete']
releaseMode: phased
inputDocuments: ['docs/index.md', 'docs/project-overview.md', 'docs/architecture.md', 'docs/source-tree-analysis.md', 'docs/development-guide.md', 'docs/api-contracts.md']
workflowType: 'prd'
documentCounts:
  briefs: 0
  research: 0
  brainstorming: 0
  projectDocs: 6
classification:
  projectType: web_app
  domain: multimedia_content_creation
  complexity: medium-high
  projectContext: brownfield
  refactorStrategy: phased
  techStack:
    backend: FastAPI + async
    concurrency: BackgroundTasks + concurrent.futures
    frontend: React + Vite
    stateManagement: Zustand + XState
    ui: Tailwind CSS + Headless UI
    dataStore: SQLite (replacing state.json)
    testing: pytest + Vitest + Playwright
    config: env vars + pathlib cross-platform
  phases:
    - 'Phase 1: FastAPI wrapper + config externalization + backend modularization'
    - 'Phase 2: React frontend rewrite + WebSocket state flow'
    - 'Phase 3: Test coverage + concurrency optimization + SQLite migration'
---

# Product Requirements Document - 视频处理

**Author:** Administrator
**Date:** 2026-05-12

## Executive Summary

Voice Studio 是一个本地优先的一站式 AI 视频配音平台，用户从原始视频到完成配音成片只需一个工作流。本 PRD 定义了一次分阶段架构重构，将现有的功能原型升级为生产级、跨平台、可扩展的应用，为团队协作、开源发布和商业化打好基础。

本次重构解决现有单体架构的五个系统性问题：3600 行单文件后端、4000 行单文件前端、硬编码 Windows 路径、零测试覆盖、以及无法扩展的线程并发模型。目标架构引入 FastAPI 异步后端、React 组件化前端、环境变量驱动的配置、全栈自动化测试、以及结构化的后台任务管理。

重构采用分阶段迁移策略以保持持续交付：第一阶段模块化后端并外部化配置，第二阶段用 React 重写前端，第三阶段补充测试覆盖并优化并发。每个阶段结束时系统均可正常使用。

### What Makes This Special

Voice Studio 将三种能力整合在一个本地优先的开源工具中，这是市面上没有的组合：

- **本地优先隐私保障** — 所有 AI 推理（语音识别、语音合成、声音克隆）在用户 GPU 上运行，数据不出本机。对比 ElevenLabs、Descript 等需要上传敏感内容的云端工具。
- **端到端全流程** — 从视频上传、转录、文案优化、语音克隆到最终视频合成（含字幕烧录），无需在多个工具间切换。
- **AI 深度集成** — CosyVoice 零样本声音克隆 + faster-whisper 转录 + LLM 文案润色，作为一个连贯的管线协同工作，而非简单拼凑。

## Project Classification

- **项目类型：** Web 应用（React SPA + FastAPI 后端）
- **领域：** 多媒体内容创作 / AI 辅助视频制作
- **复杂度：** 中高（GPU 推理、实时通信、视频处理、跨平台、全栈测试）
- **项目上下文：** 棕地重构 — 对约 8000 行代码、60+ API 端点的可运行原型进行分阶段现代化

## Success Criteria

### User Success (Developer Experience)

- **新功能开发效率：** 一天内能完成一个独立工具/功能模块的开发和上线
- **改 bug 不引入新 bug：** 核心管线有测试覆盖，改动后 CI 绿灯即可放心发布
- **新人上手：** 项目结构清晰，模块职责明确，不需要通读 3600 行代码才能改东西

### User Success (End User)

- **功能无感迁移：** 重构后所有现有功能保持不变，用户体验不退化
- **性能不降级：** 转录、克隆、合成的速度不低于当前版本

### Business Success

- **尽快开源：** 重构完成后即可在 GitHub 发布，代码质量达到开源标准
- **跨平台就绪：** 架构上消除 Windows 硬编码，首先保证 Windows 可运行，Mac 支持作为后续目标
- **可商业化基础：** 模块化架构支持未来加入多用户、权限、付费功能

### Technical Success

- **测试覆盖：** 核心管线（转录→克隆→合成）必须有自动化测试，其余模块逐步补充
- **模块化：** 后端按功能域拆分（项目管理、语音克隆、图生视频、工具集），单个模块 < 500 行
- **零硬编码路径：** 所有路径通过环境变量或配置文件指定
- **CI 可运行：** 测试套件能在 Windows 环境下自动运行

### Measurable Outcomes

- 新增一个视频编辑工具：< 1 天（含测试）
- 后端单文件 → 模块数：≥ 8 个独立模块
- 测试通过后引入回归 bug 的概率：趋近于零
- 从 clone 到首次 `python main.py` 可运行：< 10 分钟（含依赖安装）

## Product Scope

See "Project Scoping & Phased Development" section for detailed phase breakdown.

## User Journeys

### Journey 1: 内容创作者 — 小王给教程视频配音

**Opening Scene:** 小王是一个技术博主，录了一段 20 分钟的编程教程视频。他普通话带口音，想用 AI 重新配一个标准的旁白。

**Rising Action:** 他打开 Voice Studio，创建项目，上传视频。系统自动转录出字幕，他快速浏览修改了几个错别字，点击"AI 润色"让 LLM 把口语化的表达改成书面语。然后他从音色库选了"标准播音"风格，一键克隆。

**Climax:** 3 分钟后，克隆完成。他逐句试听，发现第 7 句语气不太对，点击"重新生成"单独重做了这一句。满意后点击"全部采纳"。

**Resolution:** 点击"生成视频"，系统自动合成带字幕的最终视频。小王下载成片，直接上传到 B 站。整个过程 15 分钟，没有离开过 Voice Studio。

### Journey 2: 开发者 — 小李给 Voice Studio 加一个"视频水印"工具

**Opening Scene:** 小李是一个开源贡献者，想给 Voice Studio 的工具集加一个视频水印功能。他 fork 了仓库，clone 到本地。

**Rising Action:** 他看了项目结构，发现工具集的路由在 `routes/tools.py`，业务逻辑在 `services/tools/`，前端组件在 `src/pages/Tools/`。他照着现有的"格式转换"工具的模式，新建了 `services/tools/watermark.py` 和对应的路由、React 组件。

**Climax:** 写完后他跑了一下测试：`pytest tests/api/test_tools.py` 全绿，`npm run test` 组件测试通过。他启动 dev server 在浏览器里试了一下，水印功能正常工作。

**Resolution:** 从开始到提交 PR，花了大半天。他不需要理解整个 3600 行文件，只动了 3 个文件，每个都不超过 100 行。测试给了他信心——不会破坏其他功能。

### Journey 3: 部署者 — 小张在新 Windows 机器上安装 Voice Studio

**Opening Scene:** 小张的同事推荐了 Voice Studio，他想在自己的 Windows 台式机（有 RTX 3060）上跑起来。

**Rising Action:** 他 clone 了仓库，看到 README 里的安装步骤：
1. `python -m venv .venv && .venv\Scripts\activate`
2. `pip install -r requirements.txt`
3. 复制 `.env.example` 为 `.env`，填入自己的 LLM API key
4. `python -m voice_studio`

**Climax:** 第一次启动时，系统自动检测到 CUDA 可用，提示下载 Whisper 和 CosyVoice 模型。模型下载完成后，服务启动成功。

**Resolution:** 从 clone 到浏览器打开 Voice Studio，总共不到 10 分钟（不含模型下载时间）。没有手动改任何路径配置，`.env` 里只填了 API key。

### Journey Requirements Summary

| 旅程 | 揭示的能力需求 |
|------|---------------|
| 内容创作者 | 所有现有功能必须完整保留；UI 响应速度不退化；WebSocket 进度反馈流畅 |
| 开发者 | 模块化目录结构；清晰的分层模式可复制；测试套件快速反馈；开发文档完善 |
| 部署者 | 零硬编码路径；.env 配置驱动；模型自动检测/下载提示；跨平台 pathlib |

## Web App Specific Requirements

### Project-Type Overview

Voice Studio 是一个本地运行的 SPA Web 应用，前后端分离架构。前端通过 REST API + WebSocket 与后端通信，后端负责 GPU 推理和视频处理。不需要 SEO、不需要 SSR，纯客户端渲染。

### Technical Architecture Considerations

**前端架构：**
- SPA（React + Vite），客户端路由
- 浏览器支持：Chrome（最新两个版本）
- 实时通信：WebSocket（进度推送、状态同步）
- 大文件上传：视频文件可能数百 MB，需要分片上传或流式上传
- 音频录制：浏览器 MediaRecorder API（录制声纹样本）
- 视频播放：HTML5 Video（逐句片段 + 最终成片预览）

**后端架构：**
- FastAPI（async）+ Uvicorn
- 路由按功能域拆分：projects、voices、img2vid、tools、llm
- 服务层：业务逻辑独立于路由
- 任务管理：BackgroundTasks + ThreadPoolExecutor（GPU 推理、FFmpeg）
- 文件服务：静态文件 + 流式视频响应

**前后端通信：**
- REST API：CRUD 操作、配置管理
- WebSocket：长任务进度（转录、克隆、合成）、状态变更通知
- 文件上传/下载：multipart form-data 上传，流式下载

### Implementation Considerations

**状态管理分层（前端）：**
- 全局状态（Zustand）：当前项目、LLM 配置、音色库
- 流程状态机（XState）：项目生命周期（new→processing→editing→recording→cloning→composing→done）
- 服务端状态：WebSocket 事件驱动更新
- 组件局部状态：表单输入、UI 交互

**模块拆分（后端）：**
- `routes/` — API 路由定义（thin layer）
- `services/` — 业务逻辑（projects、voice_clone、img2vid、tools、llm）
- `models/` — 数据模型和 schema（Pydantic）
- `workers/` — 后台任务（transcribe、clone、compose）
- `config/` — 配置加载、路径解析
- `utils/` — FFmpeg 封装、文件操作

**跨平台路径策略：**
- 所有路径使用 `pathlib.Path`
- 外部依赖路径通过 `.env` 配置（FFMPEG_PATH、COSYVOICE_DIR、HF_CACHE）
- 项目数据目录通过 DATA_DIR 环境变量指定
- 默认值使用相对路径（`./data/projects`）

## Project Scoping & Phased Development

### MVP Strategy & Philosophy

**MVP 方式：** 问题解决型 — Phase 1 的目标是解决"改 bug 引入新 bug"和"无法跨平台"这两个核心痛点，同时保持所有功能可用。

**资源：** 单人开发，尽快完成以推进开源。

### Phase 1: 后端重构（MVP）

**支持的用户旅程：** 内容创作者（功能不变）、部署者（配置化安装）

**Must-Have：**
- FastAPI 替换 Flask，60+ 端点功能等价迁移
- 后端拆分为 ≥8 个模块（routes/services/workers/config/utils/models）
- 所有硬编码路径改为 .env 配置
- BackgroundTasks + ThreadPoolExecutor 替换 daemon 线程
- WebSocket 进度推送保持不变
- 后端 API 测试覆盖核心管线（转录、克隆、合成）
- Windows 验证通过
- 现有前端（单 HTML）能直接对接新后端

**Nice-to-Have：**
- SQLite 替代 state.json
- 模型自动检测/下载提示

### Phase 2: 前端重写

**支持的用户旅程：** 内容创作者（体验不退化）、开发者（组件化可扩展）

**Must-Have：**
- React + Vite + Zustand + XState 完整重写
- Tailwind + Headless UI 组件库
- WebSocket 状态流独立层
- 所有现有页面/功能等价迁移
- 前端组件测试（Vitest）
- E2E 测试（Playwright）覆盖主流程

### Phase 3: 质量 & 扩展

**支持的用户旅程：** 开发者（测试信心）、部署者（跨平台）

**Must-Have：**
- 测试覆盖补全（边缘场景、错误恢复）
- Mac/Linux 跨平台验证
- GitHub 开源发布（README、LICENSE、CONTRIBUTING）
- Docker 部署方案

**Nice-to-Have：**
- 多用户 + 权限
- 插件化架构
- 商业化功能

### Risk Mitigation Strategy

**技术风险：**
- GPU 推理在 FastAPI async 环境下的兼容性 → Phase 1 初期做 spike 验证
- FFmpeg 子进程在 async 上下文中的行为 → 用 ThreadPoolExecutor 隔离
- 前端重写期间功能回归 → Phase 1 的 API 测试作为契约保障

**市场风险：**
- 开源后无人关注 → 先在中文社区（B站、掘金）发布教程视频
- 竞品追赶 → 本地优先 + 全流程一体化是护城河

**资源风险：**
- 单人开发周期长 → 分阶段交付，每阶段可用
- Phase 1 如果超时 → 优先保证核心管线（转录+克隆+合成），工具集可延后

## Functional Requirements

### 项目管理

- FR1: 用户可以创建配音项目（上传视频，指定项目名）
- FR2: 用户可以查看所有项目列表及其当前状态
- FR3: 用户可以删除项目及其所有关联数据
- FR4: 用户可以重置项目数据（保留项目壳）
- FR5: 用户可以重新上传视频替换原始素材

### 语音转录

- FR6: 系统可以从上传视频中提取音频并自动转录为逐句字幕
- FR7: 系统在转录过程中实时推送进度到前端
- FR8: 用户可以查看和编辑转录结果（修改文本、调整时间戳）
- FR9: 用户可以在多个字幕版本间切换（原始/优化/上传）

### 文案优化

- FR10: 用户可以触发 LLM 对字幕进行自动润色（去口头禅、修错别字、整合碎句）
- FR11: 用户可以上传自己的字幕文本并自动匹配到原始时间轴
- FR12: 用户可以导出字幕为 SRT/TXT/JSON 格式
- FR13: 用户可以删除/恢复单个句子（从最终合成中排除）

### 语音克隆

- FR14: 用户可以从预设音色库中选择一个声音风格
- FR15: 用户可以上传自定义声纹样本进行零样本克隆
- FR16: 用户可以保存/管理/删除自定义音色
- FR17: 系统可以逐句生成克隆语音并实时推送进度
- FR18: 用户可以逐句试听克隆结果
- FR19: 用户可以对单句重新生成克隆
- FR20: 用户可以一键采纳所有克隆结果
- FR21: 用户可以取消正在进行的克隆任务
- FR22: 系统可以从断点恢复克隆任务（跳过已完成句子）
- FR23: 用户可以手动录制单句语音替代克隆

### 视频合成

- FR24: 系统可以将逐句录音/克隆音频与视频片段合成为最终视频
- FR25: 系统在合成过程中烧录字幕（支持自定义字幕样式）
- FR26: 用户可以自定义字幕样式（字体、颜色、位置、大小）
- FR27: 用户可以预览字幕效果（静态帧 + 样式叠加）
- FR28: 用户可以下载最终合成视频
- FR29: 用户可以预览最终合成视频（在线播放）

### 图生视频

- FR30: 用户可以创建图生视频项目（上传图片序列 + 主题）
- FR31: 用户可以添加/删除/拖拽排序图片
- FR32: 系统可以通过 Vision LLM 分析图片内容并生成旁白文案
- FR33: 用户可以编辑 AI 生成的旁白文案
- FR34: 用户可以录制声纹样本用于图生视频配音
- FR35: 系统可以逐句生成 TTS 配音并支持试听
- FR36: 系统可以将图片序列 + 配音 + 字幕合成为视频（支持 Ken Burns 动画）
- FR37: 用户可以选择旁白风格（纪录片/幽默/故事/科普/产品/新闻）

### 视频编辑工具

- FR38: 用户可以上传视频到独立工具工作区
- FR39: 用户可以删除视频中的指定时间段
- FR40: 用户可以在指定位置插入视频片段
- FR41: 用户可以拼接多个视频
- FR42: 用户可以对指定时间段进行变速处理
- FR43: 用户可以替换指定时间段的音频
- FR44: 用户可以转换视频格式和分辨率（mp4/avi/mkv/mov/webm × 1080p/720p/480p/360p）
- FR45: 工具操作支持链式编辑（每次操作基于上一次结果）
- FR46: 用户可以下载工具处理结果

### 配置与 LLM 管理

- FR47: 用户可以配置多个 LLM 连接（OpenAI/Anthropic/兼容端点）
- FR48: 用户可以切换活跃的 LLM 配置
- FR49: 用户可以测试 LLM 连接是否正常
- FR50: 系统通过环境变量加载所有路径配置（FFmpeg、模型目录、数据目录）

### 开发者体验（重构新增）

- FR51: 开发者可以通过标准化模式添加新的视频编辑工具（路由+服务+测试）
- FR52: 开发者可以运行后端测试套件验证 API 正确性
- FR53: 开发者可以运行前端测试套件验证组件行为
- FR54: 开发者可以运行 E2E 测试验证完整用户流程
- FR55: 开发者可以通过 .env 文件配置所有外部依赖路径

## Non-Functional Requirements

### Performance

- 页面首次加载时间 < 3 秒（本地网络）
- API 响应时间（非 GPU 操作）< 200ms
- WebSocket 进度推送延迟 < 500ms
- 视频上传不阻塞 UI 交互
- GPU 推理性能不低于当前版本（同硬件条件下）

### Reliability

- 长任务（克隆、合成）中途失败时，已完成的部分不丢失
- 克隆任务支持断点续传（重启后从上次位置继续）
- FFmpeg 子进程失败时返回有意义的错误信息
- WebSocket 断连后自动重连并恢复状态同步

### Maintainability

- 单个模块文件不超过 500 行
- 新增一个视频编辑工具只需修改 ≤3 个文件
- 后端测试套件运行时间 < 60 秒（mock GPU）
- 代码符合 Python/TypeScript 标准 lint 规则（ruff/eslint）

### Compatibility

- Windows 10/11 完整支持（Phase 1）
- macOS 架构兼容（无 Windows 专属 API 调用）（Phase 3）
- Python 3.11+ 运行时
- Node.js 18+ 构建环境
- NVIDIA GPU（CUDA 11.8+）用于推理加速

### Security

- LLM API Key 不以明文存储在代码仓库中
- .env 文件在 .gitignore 中
- API Key 在前端展示时脱敏（只显示首尾 4 位）
