import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiGet, apiPost, apiPut, apiDelete, apiUpload } from './client'

export interface Img2VidProject {
  name: string
  stage: string
  msg: string
  theme?: string
  image_count?: number
  generate_progress?: [number, number]
}

export interface NarrationItem {
  image_idx: number
  narration: string
  analysis?: string
}

export function useImg2VidList() {
  return useQuery<Img2VidProject[]>({
    queryKey: ['img2vid'],
    queryFn: () => apiGet('/api/img2vid'),
  })
}

export function useImg2VidStatus(name: string | null) {
  return useQuery<Img2VidProject>({
    queryKey: ['img2vid', name],
    queryFn: () => apiGet(`/api/img2vid/${name}`),
    enabled: !!name,
    refetchInterval: (query) => {
      const stage = query.state.data?.stage
      return stage === 'analyzing' || stage === 'preview' || stage === 'generating' ? 2000 : false
    },
  })
}

export function useCreateImg2Vid() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async ({ name, theme, images }: { name: string; theme: string; images: File[] }) => {
      const fd = new FormData()
      fd.append('name', name)
      fd.append('theme', theme)
      images.forEach((img, i) => fd.append(`image${i}`, img))
      return apiUpload('/api/img2vid', fd)
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ['img2vid'] }),
  })
}

export function useDeleteImg2Vid() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) => apiDelete(`/api/img2vid/${name}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['img2vid'] }),
  })
}

export function useAnalyzeImg2Vid() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ name, style }: { name: string; style: string }) =>
      apiPost(`/api/img2vid/${name}/analyze`, { style }),
    onSuccess: (_, vars) => qc.invalidateQueries({ queryKey: ['img2vid', vars.name] }),
  })
}

export function useNarration(name: string | null) {
  return useQuery<{ items: NarrationItem[] }>({
    queryKey: ['img2vid', name, 'narration'],
    queryFn: () => apiGet(`/api/img2vid/${name}/narration`),
    enabled: !!name,
  })
}

export function useSaveNarration() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ name, items }: { name: string; items: NarrationItem[] }) =>
      apiPut(`/api/img2vid/${name}/narration`, { items }),
    onSuccess: (_, vars) =>
      qc.invalidateQueries({ queryKey: ['img2vid', vars.name, 'narration'] }),
  })
}
