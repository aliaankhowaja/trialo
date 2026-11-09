// Thin fetch client for the TrialO backend.
// In production nginx proxies /api; in dev Vite does the same. A VITE_API_BASE
// env (set at build time) overrides both for deployments where the API lives
// on another origin.
const BASE = import.meta.env.VITE_API_BASE || ''

export class ApiError extends Error {
  constructor(message, status, detail) {
    super(message)
    this.status = status
    this.detail = detail
  }
}

async function request(path, options = {}) {
  const response = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!response.ok) {
    let detail = null
    try {
      const body = await response.json()
      detail = typeof body.detail === 'string'
        ? body.detail
        : (body.detail ? JSON.stringify(body.detail) : JSON.stringify(body))
    } catch {
      detail = response.statusText
    }
    throw new ApiError(`Request failed (${response.status})`, response.status, detail)
  }
  return response
}

export async function getJson(path) {
  const response = await request(path)
  return response.json()
}

export async function postJson(path, body) {
  const response = await request(path, { method: 'POST', body: JSON.stringify(body) })
  return response.json()
}

export async function postBlob(path, body) {
  const response = await request(path, { method: 'POST', body: JSON.stringify(body) })
  return response.blob()
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(url)
}

export function downloadMarkdown(md, filename = 'eligibility_screening_report.md') {
  downloadBlob(new Blob([md], { type: 'text/markdown' }), filename)
}

export async function downloadPdf(payload, filename = 'eligibility_screening_report.pdf') {
  const blob = await postBlob('/api/report/pdf', payload)
  downloadBlob(blob, filename)
}

// POST + consume a Server-Sent-Events stream (EventSource cannot POST).
// onEvent receives parsed `data` objects: {type: 'meta'|'result'|'done', ...}
export async function postSse(path, body, onEvent) {
  const response = await request(path, { method: 'POST', body: JSON.stringify(body) })
  if (!response.body) throw new ApiError('Streaming not supported by this browser', 0)

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const chunks = buffer.split('\n\n')
    buffer = chunks.pop()
    for (const chunk of chunks) {
      const line = chunk.split('\n').find((l) => l.startsWith('data: '))
      if (line) onEvent(JSON.parse(line.slice(6)))
    }
  }
}
