import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useImg2VidList, useCreateImg2Vid, useDeleteImg2Vid } from '../../api/img2vid'

export default function Img2VidList() {
  const { data: projects, isLoading, error } = useImg2VidList()
  const createMut = useCreateImg2Vid()
  const deleteMut = useDeleteImg2Vid()

  const [showCreate, setShowCreate] = useState(false)
  const [newName, setNewName] = useState('')
  const [theme, setTheme] = useState('')
  const [images, setImages] = useState<File[]>([])

  if (isLoading) return <div className="text-gray-500">加载中...</div>
  if (error) return <div className="text-red-600">错误: {(error as Error).message}</div>

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!newName.trim() || images.length === 0) return
    try {
      await createMut.mutateAsync({ name: newName.trim(), theme, images })
      setShowCreate(false)
      setNewName('')
      setTheme('')
      setImages([])
    } catch (err) {
      alert(`创建失败: ${(err as Error).message}`)
    }
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-900">图生视频</h1>
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
            type="text"
            value={theme}
            onChange={(e) => setTheme(e.target.value)}
            placeholder="主题 (可选)"
            className="w-full px-3 py-2 border rounded"
          />
          <input
            type="file"
            accept="image/*"
            multiple
            onChange={(e) => setImages(Array.from(e.target.files ?? []))}
            required
          />
          <div className="text-sm text-gray-500">已选 {images.length} 张图片</div>
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
              <Link
                to={`/img2vid/${p.name}`}
                className="font-semibold text-gray-900 hover:text-blue-600"
              >
                {p.name}
              </Link>
              <button
                onClick={() => {
                  if (confirm(`确定删除项目 ${p.name} ?`)) deleteMut.mutate(p.name)
                }}
                className="text-red-600 hover:text-red-800 text-sm"
              >
                删除
              </button>
            </div>
            {p.theme && <div className="text-sm text-gray-600 mb-1">主题: {p.theme}</div>}
            <div className="text-sm text-gray-600">图片数: {p.image_count ?? 0}</div>
            <div className="text-sm text-gray-600">状态: {p.stage}</div>
            {p.msg && <div className="text-sm text-gray-500 mt-1 truncate">{p.msg}</div>}
          </div>
        ))}
      </div>
    </div>
  )
}
