import { useParams, Link } from 'react-router-dom'
import { useProjectStatus, useSentences, useOptimize, useStartClone, useCancelClone, useAcceptAllClones, useCompose, useSetStage, useVoices } from '../../api/projectDetail'
import { useWebSocket } from '../../hooks/useWebSocket'
import { useState } from 'react'

export default function ProjectDetail() {
  const { name } = useParams<{ name: string }>()
  const { data: status } = useProjectStatus(name ?? null)
  const { data: sentencesData } = useSentences(name ?? null)
  const { data: voices } = useVoices()
  const optimizeMut = useOptimize()
  const cloneMut = useStartClone()
  const cancelMut = useCancelClone()
  const acceptAllMut = useAcceptAllClones()
  const composeMut = useCompose()
  const setStageMut = useSetStage()

  useWebSocket()

  const [selectedVoice, setSelectedVoice] = useState('standard')
  const [optimizeDesc, setOptimizeDesc] = useState('')

  if (!name) return <div>缺少项目名</div>

  const stage = status?.stage || 'new'
  const progress = status?.clone_progress

  return (
    <div>
      <div className="mb-4">
        <Link to="/projects" className="text-blue-600 hover:underline">← 返回列表</Link>
      </div>
      <h1 className="text-2xl font-bold text-gray-900 mb-2">{name}</h1>

      {/* Status Bar */}
      <div className="mb-6 p-4 bg-white border rounded">
        <div className="flex justify-between items-center">
          <div>
            <span className="text-sm text-gray-600">状态: </span>
            <span className="font-medium">{stage}</span>
            {status?.msg && <span className="ml-2 text-sm text-gray-500">{status.msg}</span>}
          </div>
          {stage === 'done' && (
            <a href={`/api/project/${name}/download`} target="_blank"
              className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700">
              下载视频
            </a>
          )}
        </div>
        {progress && progress[1] > 0 && (
          <div className="mt-3">
            <div className="h-2 bg-gray-200 rounded overflow-hidden">
              <div className="h-full bg-blue-600 transition-all"
                style={{ width: `${(progress[0] / progress[1]) * 100}%` }} />
            </div>
            <div className="text-xs text-gray-500 mt-1">{progress[0]} / {progress[1]}</div>
          </div>
        )}
      </div>

      {/* Stage: Editing */}
      {(stage === 'editing' || stage === 'new') && (
        <div className="mb-6 p-4 bg-white border rounded">
          <h2 className="font-semibold mb-3">字幕编辑</h2>
          {sentencesData?.sentences && sentencesData.sentences.length > 0 ? (
            <>
              <div className="text-sm text-gray-500 mb-2">
                当前版本: {sentencesData.active} · {sentencesData.sentences.length} 句
              </div>
              <div className="max-h-64 overflow-y-auto border rounded p-2 mb-3">
                {sentencesData.sentences.slice(0, 20).map((s, i) => (
                  <div key={i} className="py-1 border-b last:border-0 text-sm">
                    <span className="text-gray-400 mr-2">{i + 1}.</span>
                    {s.text}
                  </div>
                ))}
                {sentencesData.sentences.length > 20 && (
                  <div className="text-gray-400 text-sm py-1">...还有 {sentencesData.sentences.length - 20} 句</div>
                )}
              </div>
              <div className="flex gap-3 items-end">
                <div className="flex-1">
                  <input value={optimizeDesc} onChange={(e) => setOptimizeDesc(e.target.value)}
                    placeholder="描述视频内容（可选，帮助 AI 更好地润色）"
                    className="w-full px-3 py-2 border rounded text-sm" />
                </div>
                <button onClick={() => optimizeMut.mutate({ name, version: sentencesData.active, description: optimizeDesc })}
                  disabled={optimizeMut.isPending}
                  className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50 text-sm">
                  AI 润色
                </button>
                <button onClick={() => setStageMut.mutate({ name, stage: 'recording', version: sentencesData.active })}
                  className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm">
                  进入录音
                </button>
              </div>
            </>
          ) : (
            <div className="text-gray-500">等待转录完成...</div>
          )}
        </div>
      )}

      {/* Stage: Recording / Cloning */}
      {(stage === 'recording' || stage === 'cloning') && (
        <div className="mb-6 p-4 bg-white border rounded">
          <h2 className="font-semibold mb-3">语音克隆</h2>
          {stage === 'cloning' ? (
            <div>
              <div className="text-sm text-gray-600 mb-2">克隆进行中...</div>
              <button onClick={() => cancelMut.mutate(name)}
                className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 text-sm">
                取消克隆
              </button>
            </div>
          ) : (
            <div className="space-y-3">
              <div>
                <label className="text-sm text-gray-600 block mb-1">选择音色</label>
                <select value={selectedVoice} onChange={(e) => setSelectedVoice(e.target.value)}
                  className="px-3 py-2 border rounded w-full max-w-xs">
                  {voices?.map((v: any) => (
                    <option key={v.id} value={v.id}>{v.name} - {v.desc}</option>
                  ))}
                </select>
              </div>
              <div className="flex gap-3">
                <button onClick={() => cloneMut.mutate({ name, voiceId: selectedVoice })}
                  disabled={cloneMut.isPending}
                  className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 text-sm">
                  开始克隆
                </button>
                <button onClick={() => acceptAllMut.mutate(name)}
                  className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 text-sm">
                  全部采纳
                </button>
                <button onClick={() => composeMut.mutate(name)}
                  className="px-4 py-2 bg-orange-600 text-white rounded hover:bg-orange-700 text-sm">
                  生成视频
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Stage: Composing */}
      {stage === 'composing' && (
        <div className="mb-6 p-4 bg-white border rounded">
          <h2 className="font-semibold mb-3">视频合成中</h2>
          <div className="text-sm text-gray-600">{status?.msg || '处理中...'}</div>
        </div>
      )}

      {/* Stage: Done */}
      {stage === 'done' && (
        <div className="mb-6 p-4 bg-white border rounded">
          <h2 className="font-semibold mb-3">完成</h2>
          <div className="flex gap-3">
            <a href={`/api/project/${name}/final-video`} target="_blank"
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm">
              预览视频
            </a>
            <a href={`/api/project/${name}/download`} target="_blank"
              className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 text-sm">
              下载
            </a>
            <button onClick={() => setStageMut.mutate({ name, stage: 'editing' })}
              className="px-4 py-2 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 text-sm">
              返回编辑
            </button>
          </div>
        </div>
      )}

      {/* Video Preview */}
      {stage !== 'new' && stage !== 'processing' && (
        <div className="mb-6 p-4 bg-white border rounded">
          <h2 className="font-semibold mb-3">原始视频</h2>
          <video controls className="w-full max-w-2xl rounded" preload="metadata">
            <source src={`/api/project/${name}/video`} type="video/mp4" />
          </video>
        </div>
      )}
    </div>
  )
}
