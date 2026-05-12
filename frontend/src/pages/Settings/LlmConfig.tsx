import { useState, useEffect } from 'react'
import { useLlmConfig, useSaveLlmConfig, useTestLlm, LlmConfigEntry, LlmConfigData } from '../../api/llm'

const EMPTY_CONFIG: LlmConfigEntry = { name: '', provider: 'openai', api_key: '', model: '', base_url: '' }

export default function LlmConfig() {
  const { data, isLoading } = useLlmConfig()
  const saveMut = useSaveLlmConfig()
  const testMut = useTestLlm()

  const [configs, setConfigs] = useState<LlmConfigEntry[]>([])
  const [activeIdx, setActiveIdx] = useState(0)

  useEffect(() => {
    if (data) {
      setConfigs(data.configs)
      setActiveIdx(data.active_idx)
    }
  }, [data])

  if (isLoading) return <div className="text-gray-500">加载中...</div>

  const handleSave = async () => {
    const payload: LlmConfigData = { configs, active_idx: activeIdx }
    try {
      await saveMut.mutateAsync(payload)
      alert('保存成功')
    } catch (err) {
      alert(`保存失败: ${(err as Error).message}`)
    }
  }

  const handleTest = async (cfg: LlmConfigEntry) => {
    try {
      const result = await testMut.mutateAsync(cfg)
      if (result.ok) alert(`连接成功: ${result.result}`)
      else alert(`连接失败: ${result.error}`)
    } catch (err) {
      alert(`测试失败: ${(err as Error).message}`)
    }
  }

  const addConfig = () => setConfigs([...configs, { ...EMPTY_CONFIG, name: `配置 ${configs.length + 1}` }])
  const removeConfig = (idx: number) => {
    setConfigs(configs.filter((_, i) => i !== idx))
    if (activeIdx >= configs.length - 1) setActiveIdx(Math.max(0, configs.length - 2))
  }
  const updateConfig = (idx: number, field: keyof LlmConfigEntry, value: string) => {
    setConfigs(configs.map((c, i) => (i === idx ? { ...c, [field]: value } : c)))
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-900">LLM 配置</h1>
        <div className="flex gap-2">
          <button onClick={addConfig} className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
            添加配置
          </button>
          <button onClick={handleSave} disabled={saveMut.isPending}
            className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50">
            {saveMut.isPending ? '保存中...' : '保存'}
          </button>
        </div>
      </div>

      <div className="space-y-4">
        {configs.map((cfg, idx) => (
          <div key={idx} className={`p-4 bg-white border rounded ${idx === activeIdx ? 'border-blue-500 ring-1 ring-blue-200' : ''}`}>
            <div className="flex justify-between items-center mb-3">
              <div className="flex items-center gap-3">
                <input type="radio" name="active" checked={idx === activeIdx}
                  onChange={() => setActiveIdx(idx)} className="w-4 h-4" />
                <span className="text-sm text-gray-500">{idx === activeIdx ? '当前使用' : '备用'}</span>
              </div>
              <div className="flex gap-2">
                <button onClick={() => handleTest(cfg)} disabled={testMut.isPending}
                  className="px-3 py-1 text-sm bg-gray-100 border rounded hover:bg-gray-200 disabled:opacity-50">
                  测试
                </button>
                <button onClick={() => removeConfig(idx)}
                  className="px-3 py-1 text-sm text-red-600 border border-red-200 rounded hover:bg-red-50">
                  删除
                </button>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <input value={cfg.name} onChange={(e) => updateConfig(idx, 'name', e.target.value)}
                placeholder="配置名称" className="px-3 py-2 border rounded" />
              <select value={cfg.provider} onChange={(e) => updateConfig(idx, 'provider', e.target.value)}
                className="px-3 py-2 border rounded">
                <option value="openai">OpenAI 兼容</option>
                <option value="anthropic">Anthropic</option>
              </select>
              <input value={cfg.api_key} onChange={(e) => updateConfig(idx, 'api_key', e.target.value)}
                placeholder="API Key" type="password" className="px-3 py-2 border rounded" />
              <input value={cfg.model} onChange={(e) => updateConfig(idx, 'model', e.target.value)}
                placeholder="模型名 (如 gpt-4o-mini)" className="px-3 py-2 border rounded" />
              <input value={cfg.base_url} onChange={(e) => updateConfig(idx, 'base_url', e.target.value)}
                placeholder="Base URL (可选)" className="px-3 py-2 border rounded md:col-span-2" />
            </div>
          </div>
        ))}
        {configs.length === 0 && (
          <div className="text-gray-500 text-center py-8">还没有配置，点击"添加配置"开始</div>
        )}
      </div>
    </div>
  )
}
