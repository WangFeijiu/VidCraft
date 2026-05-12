import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiGet, apiPost, apiPut, apiUpload } from './client'

export interface Sentence {
  text: string
  start: number
  end: number
}

export interface SentencesData {
  active: string
  versions: Record<string, Sentence[]>
  sentences: Sentence[]
}

export function useProjectStatus(name: string | null) {
  return useQuery<any>({
    queryKey: ['project', name],
    queryFn: () => apiGet(`/api/project/${name}`),
    enabled: !!name,
    refetchInterval: (query) => {
      const stage = query.state.data?.stage
      return stage === 'processing' || stage === 'cloning' || stage === 'composing' ? 2000 : false
    },
  })
}

export function useSentences(name: string | null) {
  return useQuery<SentencesData>({
    queryKey: ['project', name, 'sentences'],
    queryFn: () => apiGet(`/api/project/${name}/sentences`),
    enabled: !!name,
  })
}

export function useSaveSentences() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ name, version, sentences }: { name: string; version: string; sentences: Sentence[] }) =>
      apiPut(`/api/project/${name}/sentences`, { version, sentences, clear_after: false }),
    onSuccess: (_, vars) => qc.invalidateQueries({ queryKey: ['project', vars.name, 'sentences'] }),
  })
}

export function useOptimize() {
  return useMutation({
    mutationFn: ({ name, version, description }: { name: string; version: string; description: string }) =>
      apiPost(`/api/project/${name}/optimize`, { version, description }),
  })
}

export function useStartClone() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async ({ name, voiceId, promptText }: { name: string; voiceId: string; promptText?: string }) => {
      const fd = new FormData()
      fd.append('voice_id', voiceId)
      fd.append('prompt_text', promptText || '')
      return apiUpload(`/api/project/${name}/voice-clone`, fd)
    },
    onSuccess: (_, vars) => qc.invalidateQueries({ queryKey: ['project', vars.name] }),
  })
}

export function useCancelClone() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) => apiPost(`/api/project/${name}/cancel-clone`),
    onSuccess: (_, name) => qc.invalidateQueries({ queryKey: ['project', name] }),
  })
}

export function useAcceptAllClones() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) => apiPost(`/api/project/${name}/accept-all-clones`),
    onSuccess: (_, name) => qc.invalidateQueries({ queryKey: ['project', name] }),
  })
}

export function useCompose() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) => apiPost(`/api/project/${name}/compose`),
    onSuccess: (_, name) => qc.invalidateQueries({ queryKey: ['project', name] }),
  })
}

export function useSetStage() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ name, stage, version }: { name: string; stage: string; version?: string }) =>
      apiPut(`/api/project/${name}/stage`, { stage, version }),
    onSuccess: (_, vars) => qc.invalidateQueries({ queryKey: ['project', vars.name] }),
  })
}

export function useVoices() {
  return useQuery<any[]>({
    queryKey: ['voices'],
    queryFn: () => apiGet('/api/voices'),
  })
}
