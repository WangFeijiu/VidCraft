const BASE_URL = ''

export async function apiGet<T>(path: string): Promise<T> {
  const resp = await fetch(`${BASE_URL}${path}`)
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ error: `HTTP ${resp.status}` }))
    throw new Error(err.error || err.detail || `HTTP ${resp.status}`)
  }
  return resp.json()
}

export async function apiPost<T>(path: string, body?: unknown): Promise<T> {
  const resp = await fetch(`${BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ error: `HTTP ${resp.status}` }))
    throw new Error(err.error || err.detail || `HTTP ${resp.status}`)
  }
  return resp.json()
}

export async function apiPut<T>(path: string, body: unknown): Promise<T> {
  const resp = await fetch(`${BASE_URL}${path}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ error: `HTTP ${resp.status}` }))
    throw new Error(err.error || err.detail || `HTTP ${resp.status}`)
  }
  return resp.json()
}

export async function apiDelete<T>(path: string): Promise<T> {
  const resp = await fetch(`${BASE_URL}${path}`, { method: 'DELETE' })
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ error: `HTTP ${resp.status}` }))
    throw new Error(err.error || err.detail || `HTTP ${resp.status}`)
  }
  return resp.json()
}

export async function apiUpload<T>(path: string, formData: FormData): Promise<T> {
  const resp = await fetch(`${BASE_URL}${path}`, {
    method: 'POST',
    body: formData,
  })
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ error: `HTTP ${resp.status}` }))
    throw new Error(err.error || err.detail || `HTTP ${resp.status}`)
  }
  return resp.json()
}
