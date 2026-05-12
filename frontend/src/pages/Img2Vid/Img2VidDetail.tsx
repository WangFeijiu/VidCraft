import { useParams, Link } from 'react-router-dom'
import { useState, useEffect } from 'react'
import {
  useImg2VidStatus,
  useAnalyzeImg2Vid,
  useNarration,
  useSaveNarration,
  NarrationItem,
} from '../../api/img2vid'

const STYLES = [
  { id: 'documentary', name: '纪录片' },
  { id: 'humor', name: '幽默风趣' },
  { id: 'story', name: '故事讲述' },
  { id: 'educational', name: '科普解说' },
  { id: 'product', name: '产品介绍' },
  { id: 'news', name: '新闻报道' },
]

export default function Img2VidDetail() {
  const { name } = useParams<{ name: string }>()
  const { data: status } = useImg2VidStatus(name ?? null)
  const { data: narration } = useNarration(name ?? null)
  const analyzeMut = useAnalyzeImg2Vid()
  const saveMut = useSaveNarration()

  const [style, setStyle] = useState('documentary')
  const [items, setItems] = useState<NarrationItem[]>([])

  useEffect(() => {
    if (narration?.items) setItems(narration.items)
  }, [narration])

  if (!name) return <div>缺少项目名</div>

  const handleAnalyze = async () => {
    try {
      await analyzeMut.mutateAsync({ name, style })
    } catch (err) {
      alert(`启动失败: ${(err as Error).message}`)
    }
  }

  const handleSave = async () => {
    try {
      await saveMut.mutateAsync({ name, items })
      alert('已保存')
    } catch (err) {
      alert(`保存失败: ${(err as Error).message}`)
    }
  }

  const updateItem = (idx: number, text: string) => {
    setItems((prev) => prev.map((it, i) => (i === idx ? { ...it, narration: text } : it)))
  }

  return (
    <div>
      <div className="mb-4">
        <Link to="/img2vid" className="text-blue-600 hover:underline">← 返回列表</Link>
      </div>
      <h1 className="text-2xl font-bold text-gray-900 mb-4">{name}</h1>

      <div className="mb-6 p-4 bg-white border rounded">
        <div className="mb-2 text-sm text-gray-600">
          状态: <span className="font-medium">{status?.stage}</span>
          {status?.msg && <span className="ml-2">· {status.msg}</span>}
        </div>
        {status?.generate_progress && (
          <div className="mt-2">
            <div className="h-2 bg-gray-200 rounded overflow-hidden">
              <div
                className="h-full bg-blue-600 transition-all"
                style={{
                  width: `${
                    status.generate_progress[1] > 0
                      ? (status.generate_progress[0] / status.generate_progress[1]) * 100
                      : 0
                  }%`,
                }}
              />
            </div>
            <div className="text-xs text-gray-500 mt-1">
              {status.generate_progress[0]} / {status.generate_progress[1]}
            </div>
          </div>
        )}
      </div>

      <div className="mb-6 p-4 bg-white border rounded">
        <h2 className="font-semibold mb-3">AI 分析图片 + 生成旁白</h2>
        <div className="flex gap-3 items-center">
          <select
            value={style}
            onChange={(e) => setStyle(e.target.value)}
            className="px-3 py-2 border rounded"
          >
            {STYLES.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name}
              </option>
            ))}
          </select>
          <button
            onClick={handleAnalyze}
            disabled={analyzeMut.isPending || status?.stage === 'analyzing'}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
          >
            {status?.stage === 'analyzing' ? '分析中...' : '开始分析'}
          </button>
        </div>
      </div>

      {items.length > 0 && (
        <div className="mb-6 p-4 bg-white border rounded">
          <div className="flex justify-between items-center mb-3">
            <h2 className="font-semibold">旁白编辑</h2>
            <button
              onClick={handleSave}
              disabled={saveMut.isPending}
              className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
            >
              {saveMut.isPending ? '保存中...' : '保存'}
            </button>
          </div>
          <div className="space-y-3">
            {items.map((it, idx) => (
              <div key={idx} className="border-b pb-3">
                <div className="text-sm text-gray-500 mb-1">图片 {it.image_idx + 1}</div>
                {it.analysis && (
                  <div className="text-xs text-gray-400 mb-2 italic">分析: {it.analysis}</div>
                )}
                <textarea
                  value={it.narration}
                  onChange={(e) => updateItem(idx, e.target.value)}
                  rows={2}
                  className="w-full px-3 py-2 border rounded text-sm"
                  placeholder="旁白文字"
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
