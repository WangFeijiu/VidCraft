import { useState } from 'react'
import { useToolList, useUploadTool, useDeleteTool, useToolDelete, useToolConvert, useToolSpeedup } from '../../api/tools'

export default function ToolList() {
  const { data: sessions, isLoading } = useToolList()
  const uploadMut = useUploadTool()
  const deleteMut = useDeleteTool()
  const editDeleteMut = useToolDelete()
  const convertMut = useToolConvert()
  const speedupMut = useToolSpeedup()

  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [activeSid, setActiveSid] = useState<string | null>(null)
  const [deleteRanges, setDeleteRanges] = useState('')
  const [convertFmt, setConvertFmt] = useState('mp4')
  const [convertRes, setConvertRes] = useState('original')
  const [speedStart, setSpeedStart] = useState('')
  const [speedEnd, setSpeedEnd] = useState('')
  const [speedRate, setSpeedRate] = useState('2')

  if (isLoading) return <div className="text-gray-500">加载中...</div>

  const handleUpload = async () => {
    if (!videoFile) return
    try {
      const result = await uploadMut.mutateAsync(videoFile)
      setActiveSid(result.sid)
      setVideoFile(null)
    } catch (err) {
      alert(`上传失败: ${(err as Error).message}`)
    }
  }

  const handleDelete = async (sid: string) => {
    if (!deleteRanges.trim()) return alert('请输入时间段')
    try {
      await editDeleteMut.mutateAsync({ sid, ranges: deleteRanges })
      setDeleteRanges('')
    } catch (err) {
      alert(`删除失败: ${(err as Error).message}`)
    }
  }

  const handleConvert = async (sid: string) => {
    try {
      await convertMut.mutateAsync({ sid, format: convertFmt, resolution: convertRes })
    } catch (err) {
      alert(`转换失败: ${(err as Error).message}`)
    }
  }

  const handleSpeedup = async (sid: string) => {
    const start = parseFloat(speedStart)
    const end = parseFloat(speedEnd)
    const rate = parseFloat(speedRate)
    if (isNaN(start) || isNaN(end) || isNaN(rate)) return alert('请输入有效数字')
    try {
      await speedupMut.mutateAsync({ sid, start, end, rate })
    } catch (err) {
      alert(`变速失败: ${(err as Error).message}`)
    }
  }

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900 mb-6">视频工具</h1>

      <div className="mb-6 p-4 bg-white border rounded">
        <h2 className="font-semibold mb-3">上传视频</h2>
        <div className="flex gap-3 items-center">
          <input type="file" accept="video/*" onChange={(e) => setVideoFile(e.target.files?.[0] ?? null)} />
          <button onClick={handleUpload} disabled={!videoFile || uploadMut.isPending}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50">
            {uploadMut.isPending ? '上传中...' : '上传'}
          </button>
        </div>
      </div>

      <div className="space-y-4">
        {sessions?.length === 0 && <div className="text-gray-500">还没有视频，请先上传</div>}
        {sessions?.map((s) => (
          <div key={s.sid} className={`p-4 bg-white border rounded ${activeSid === s.sid ? 'border-blue-500' : ''}`}>
            <div className="flex justify-between items-start mb-3">
              <div>
                <span className="font-semibold cursor-pointer hover:text-blue-600"
                  onClick={() => setActiveSid(activeSid === s.sid ? null : s.sid)}>
                  {s.filename}
                </span>
                <span className="ml-3 text-sm text-gray-500">状态: {s.stage}</span>
                {s.msg && <span className="ml-2 text-sm text-gray-400">{s.msg}</span>}
              </div>
              <div className="flex gap-2">
                <a href={`/api/tool/${s.sid}/download`} target="_blank"
                  className="px-3 py-1 text-sm bg-green-100 border border-green-300 rounded hover:bg-green-200">
                  下载
                </a>
                <button onClick={() => { if (confirm('确定删除?')) deleteMut.mutate(s.sid) }}
                  className="px-3 py-1 text-sm text-red-600 border border-red-200 rounded hover:bg-red-50">
                  删除
                </button>
              </div>
            </div>

            {activeSid === s.sid && (
              <div className="mt-3 pt-3 border-t space-y-4">
                <div className="flex gap-2 items-end">
                  <div className="flex-1">
                    <label className="text-sm text-gray-600">删除时间段</label>
                    <input value={deleteRanges} onChange={(e) => setDeleteRanges(e.target.value)}
                      placeholder="00:10-00:15, 01:20-01:25" className="w-full px-3 py-2 border rounded text-sm" />
                  </div>
                  <button onClick={() => handleDelete(s.sid)}
                    className="px-3 py-2 bg-red-600 text-white rounded text-sm hover:bg-red-700">
                    裁剪
                  </button>
                </div>

                <div className="flex gap-2 items-end">
                  <div>
                    <label className="text-sm text-gray-600">格式</label>
                    <select value={convertFmt} onChange={(e) => setConvertFmt(e.target.value)}
                      className="px-3 py-2 border rounded text-sm">
                      <option value="mp4">MP4</option>
                      <option value="avi">AVI</option>
                      <option value="mkv">MKV</option>
                      <option value="webm">WebM</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-sm text-gray-600">分辨率</label>
                    <select value={convertRes} onChange={(e) => setConvertRes(e.target.value)}
                      className="px-3 py-2 border rounded text-sm">
                      <option value="original">原始</option>
                      <option value="1080p">1080p</option>
                      <option value="720p">720p</option>
                      <option value="480p">480p</option>
                    </select>
                  </div>
                  <button onClick={() => handleConvert(s.sid)}
                    className="px-3 py-2 bg-purple-600 text-white rounded text-sm hover:bg-purple-700">
                    转换
                  </button>
                </div>

                <div className="flex gap-2 items-end">
                  <div>
                    <label className="text-sm text-gray-600">起始(秒)</label>
                    <input value={speedStart} onChange={(e) => setSpeedStart(e.target.value)}
                      placeholder="10" className="w-20 px-3 py-2 border rounded text-sm" />
                  </div>
                  <div>
                    <label className="text-sm text-gray-600">结束(秒)</label>
                    <input value={speedEnd} onChange={(e) => setSpeedEnd(e.target.value)}
                      placeholder="20" className="w-20 px-3 py-2 border rounded text-sm" />
                  </div>
                  <div>
                    <label className="text-sm text-gray-600">倍速</label>
                    <input value={speedRate} onChange={(e) => setSpeedRate(e.target.value)}
                      placeholder="2" className="w-16 px-3 py-2 border rounded text-sm" />
                  </div>
                  <button onClick={() => handleSpeedup(s.sid)}
                    className="px-3 py-2 bg-orange-600 text-white rounded text-sm hover:bg-orange-700">
                    变速
                  </button>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
