import { Routes, Route, Link, Navigate } from 'react-router-dom'
import ProjectList from './pages/Projects/ProjectList'
import Img2VidList from './pages/Img2Vid/Img2VidList'
import ToolList from './pages/Tools/ToolList'
import LlmConfig from './pages/Settings/LlmConfig'

export default function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white border-b border-gray-200 px-6 py-3 flex gap-6">
        <span className="font-bold text-gray-900">Voice Studio</span>
        <Link to="/projects" className="text-gray-700 hover:text-blue-600">配音项目</Link>
        <Link to="/img2vid" className="text-gray-700 hover:text-blue-600">图生视频</Link>
        <Link to="/tools" className="text-gray-700 hover:text-blue-600">视频工具</Link>
        <Link to="/settings" className="text-gray-700 hover:text-blue-600">设置</Link>
      </nav>
      <main className="p-6">
        <Routes>
          <Route path="/" element={<Navigate to="/projects" replace />} />
          <Route path="/projects" element={<ProjectList />} />
          <Route path="/img2vid" element={<Img2VidList />} />
          <Route path="/tools" element={<ToolList />} />
          <Route path="/settings" element={<LlmConfig />} />
        </Routes>
      </main>
    </div>
  )
}
