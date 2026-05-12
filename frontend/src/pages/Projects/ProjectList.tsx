import { useState } from 'react'
import { useProjects, useCreateProject, useDeleteProject } from '../../api/projects'

export default function ProjectList() {
  const { data: projects, isLoading, error } = useProjects()
  const createMut = useCreateProject()
  const deleteMut = useDeleteProject()

  const [showCreate, setShowCreate] = useState(false)
  const [newName, setNewName] = useState('')
  const [videoFile, setVideoFile] = useState<File | null>(null)

  if (isLoading) return <div className="text-gray-500">加载中...</div>
  if (error) return <div className="text-red-600">错误: {(error as Error).message}</div>

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!newName.trim() || !videoFile) return
    try {
      await createMut.mutateAsync({ name: newName.trim(), video: videoFile })
      setShowCreate(false)
      setNewName('')
      setVideoFile(null)
    } catch (err) {
      alert(`创建失败: ${(err as Error).message}`)
    }
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-900">配音项目</h1>
        <button
          onClick={() => setShowCreate(!showCreate)}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          {showCreate ? '取消' : '新建项目'}
        </button>
      </div>

      {showCreate && (
        <form onSubmit={handleCreate} className="mb-6 p-4 bg-white border rounded space-y-3">
          <input
            type="text"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder="项目名"
            className="w-full px-3 py-2 border rounded"
            required
          />
          <input
            type="file"
            accept="video/*"
            onChange={(e) => setVideoFile(e.target.files?.[0] ?? null)}
            required
          />
          <button
            type="submit"
            disabled={createMut.isPending}
            className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
          >
            {createMut.isPending ? '上传中...' : '创建'}
          </button>
        </form>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {projects?.length === 0 && (
          <div className="col-span-full text-gray-500">还没有项目，点击右上角新建</div>
        )}
        {projects?.map((p) => (
          <div key={p.name} className="bg-white p-4 rounded border">
            <div className="flex justify-between items-start mb-2">
              <h3 className="font-semibold text-gray-900">{p.name}</h3>
              <button
                onClick={() => {
                  if (confirm(`确定删除项目 ${p.name} ?`)) deleteMut.mutate(p.name)
                }}
                className="text-red-600 hover:text-red-800 text-sm"
              >
                删除
              </button>
            </div>
            <div className="text-sm text-gray-600">状态: {p.stage}</div>
            {p.msg && <div className="text-sm text-gray-500 mt-1">{p.msg}</div>}
          </div>
        ))}
      </div>
    </div>
  )
}
