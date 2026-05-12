import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiGet, apiPut, apiPost } from './client'

export interface LlmConfigEntry {
  name: string
  provider: string
  api_key: string
  model: string
  base_url: string
}

export interface LlmConfigData {
  configs: LlmConfigEntry[]
  active_idx: number
}

export function useLlmConfig() {
  return useQuery<LlmConfigData>({
    queryKey: ['llm-config'],
    queryFn: () => apiGet('/api/llm-config'),
  })
}

export function useSaveLlmConfig() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (data: LlmConfigData) => apiPut('/api/llm-config', data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['llm-config'] }),
  })
}

export function useTestLlm() {
  return useMutation<{ ok: boolean; result?: string; error?: string }, Error, LlmConfigEntry>({
    mutationFn: (config) => apiPost('/api/llm-test', { config }),
  })
}
