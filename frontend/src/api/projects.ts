import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiGet, apiDelete, apiUpload } from './client'

export interface Project {
  name: string
  stage: string
  msg: string
  recorded?: number
}

export function useProjects() {
  return useQuery<Project[]>({
    queryKey: ['projects'],
    queryFn: () => apiGet('/api/projects'),
  })
}

export function useCreateProject() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async ({ name, video }: { name: string; video: File }) => {
      const fd = new FormData()
      fd.append('name', name)
      fd.append('video', video)
      return apiUpload('/api/projects', fd)
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ['projects'] }),
  })
}

export function useDeleteProject() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) => apiDelete(`/api/project/${name}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['projects'] }),
  })
}
