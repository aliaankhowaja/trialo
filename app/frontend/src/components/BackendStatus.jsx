import { useEffect, useState } from 'react'
import { getJson } from '../api'

// Connectivity probe with a hard timeout so a half-started Docker stack can
// never leave the UI hanging silently.
function healthOnce(timeoutMs = 8000) {
  return new Promise((resolve, reject) => {
    fetch('/api/health')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json()
      })
      .then(resolve)
      .catch(reject)
    setTimeout(() => reject(new Error('timeout')), timeoutMs)
  })
}

export default function BackendStatus({ onReady }) {
  const [state, setState] = useState('checking') // checking | ok | down
  const [detail, setDetail] = useState(null)

  const probe = (attempt = 0) => {
    setState('checking')
    healthOnce()
      .then((info) => {
        setState('ok')
        onReady(info)
      })
      .catch((err) => {
        setDetail(err.message || 'connection failed')
        setState('down')
        // The backend image can take minutes to build/install torch; keep
        // probing instead of leaving a dead screen.
        setTimeout(() => probe(attempt + 1), attempt < 20 ? 3000 : 10000)
      })
  }

  useEffect(() => probe(), []) // eslint-disable-line react-hooks/exhaustive-deps

  if (state === 'ok') return null

  if (state === 'checking') {
    return (
      <div className="panel backend-status">
        <div className="loading">
          <div className="spinner" />
          Connecting to the screening backend…
        </div>
      </div>
    )
  }

  return (
    <div className="panel backend-status down">
      <h2>Backend not answering (retrying every few seconds…)</h2>
      <p className="hint">
        Last error: <code>{detail}</code>
      </p>
      <p className="hint">
        The UI is up, but the FastAPI backend is not reachable through the
        webserver's <code>/api</code> proxy. If you just started the stack, the
        backend image may still be installing its dependencies (torch ≈ several
        GB) — that is normal for the first <code>docker compose up --build</code>.
        Check it with:
      </p>
      <pre className="note-preview">{`docker compose ps
docker compose logs -f backend     # watch the build / startup
curl http://localhost:8000/api/health`}</pre>
      <p className="hint">
        If the backend is <em>built</em> but logs show an error, common causes are:{' '}
        a missing <code>./src</code> or <code>./data</code> mount, a checkpoint
        path that does not exist on the host (<code>TRIALO_CHECKPOINT_DIR</code>),
        or a failing Hugging Face download (gated Gemma needs a valid{' '}
        <code>HF_TOKEN</code>).
      </p>
    </div>
  )
}
