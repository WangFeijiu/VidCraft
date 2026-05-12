import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiGet, apiPost, apiDelete, apiUpload } from './client'

export interface ToolSession {
  sid: string
  filename: string
  stage: string
  duration?: number
  msg?: string
}

export function useToolList() {
  return useQuery<ToolSession[]>({
    queryKey: ['tools'],
    queryFn: () => apiGet('/api/tool/list'),
  })
}

export function useToolState(sid: string | null) {
  return useQuery<ToolSession>({
    queryKey: ['tool', sid],
    queryFn: () => apiGet(`/api/tool/${sid}/state`),
    enabled: !!sid,
    refetchInterval: (query) => {
      const stage = query.state.data?.stage
      return stage === 'processing' || stage === 'editing' || stage === 'converting' ? 2000 : false
    },
  })
}

export function useUploadTool() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (video: File) => {
      const fd = new FormData()
      fd.append('video', video)
      return apiUpload<{ sid: string; filename: string }>('/api/tool/upload', fd)
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ['tools'] }),
  })
}

export function useDeleteTool() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (sid: string) => apiDelete(`/api/tool/${sid}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['tools'] }),
  })
}

export function useToolDelete() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ sid, ranges }: { sid: string; ranges: string }) =>
      apiPost(`/api/tool/${sid}/edit/delete`, { ranges }),
    onSuccess: (_, vars) => qc.invalidateQueries({ queryKey: ['tool', vars.sid] }),
  })
}

export function useToolConvert() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ sid, format, resolution }: { sid: string; format: string; resolution: string }) =>
      apiPost(`/api/tool/${sid}/convert`, { format, resolution }),
    onSuccess: (_, vars) => qc.invalidateQueries({ queryKey: ['tool', vars.sid] }),
  })
}

export function useToolSpeedup() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ sid, start, end, rate }: { sid: string; start: number; end: number; rate: number }) =>
      apiPost(`/api/tool/${sid}/edit/speedup`, { start, end, rate }),
    onSuccess: (_, vars) => qc.invalidateQueries({ queryKey: ['tool', vars.sid] }),
  })
}
