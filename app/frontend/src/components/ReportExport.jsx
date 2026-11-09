import { useState } from 'react'
import { downloadMarkdown, downloadPdf, postJson } from '../api'

// Step 4: build the readable screening report and export it as Markdown,
// printable HTML, or server-rendered PDF.
export default function ReportExport({ patient, trial, results, benchmark }) {
  const [markdown, setMarkdown] = useState(null)
  const [html, setHtml] = useState(null)
  const [busy, setBusy] = useState(null)
  const [error, setError] = useState(null)

  const payload = () => ({
    patient_id: patient.patient_id,
    trial_title: trial.title || 'Custom criterion set',
    meta: results?.meta || {},
    results: results?.results || [],
    sentences: results?.meta?.note
      ? results.meta.note.split('\n').map((line) => line.replace(/^\s*\d+\.\s?/, ''))
      : [],
    benchmark_snapshot: benchmark,
  })

  const build = async () => {
    setBusy('build')
    setError(null)
    try {
      const [mdResult] = await Promise.all([
        postJson('/api/report', payload()),
      ])
      setMarkdown(mdResult.markdown)
    } catch (err) {
      setError(err.detail || err.message)
    } finally {
      setBusy(null)
    }
  }

  const previewHtml = async () => {
    setBusy('html')
    setError(null)
    try {
      const response = await fetch('/api/report/html', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload()),
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      setHtml(await response.text())
    } catch (err) {
      setError(err.detail || err.message)
    } finally {
      setBusy(null)
    }
  }

  const exportPdf = async () => {
    setBusy('pdf')
    setError(null)
    try {
      await downloadPdf(payload())
    } catch (err) {
      setError(err.detail || err.message)
    } finally {
      setBusy(null)
    }
  }

  const exportMd = async () => {
    if (!markdown) await build()
    if (markdown) downloadMarkdown(markdown)
  }

  if (!results || !results.results.length) {
    return (
      <div className="panel">
        <h2>Export report</h2>
        <p className="hint">Run a screening first (step 3).</p>
      </div>
    )
  }

  return (
    <>
      <div className="panel">
        <h2>Export eligibility screening report</h2>
        <p className="hint">
          The report includes the run configuration, an aggregate verdict table,
          per-criterion verdicts with exactly quoted evidence, the decode benchmark
          snapshot (if you ran one), and the educational-use disclaimer on every page.
        </p>
        <div className="row">
          <button className="primary" disabled={busy} onClick={build}>
            {busy === 'build' ? 'Building…' : markdown ? 'Rebuild report' : 'Build report'}
          </button>
          <button className="secondary" disabled={!markdown || busy} onClick={exportMd}>
            Download Markdown
          </button>
          <button className="secondary" disabled={busy} onClick={previewHtml}>
            {busy === 'html' ? 'Rendering…' : 'Printable HTML preview'}
          </button>
          <button className="secondary" disabled={busy} onClick={exportPdf}>
            {busy === 'pdf' ? 'Rendering PDF…' : 'Download PDF'}
          </button>
        </div>
        {error && <div className="error-box">{error}</div>}
      </div>

      {html && (
        <div className="panel">
          <h2>Printable preview (use the browser's print dialog for a PDF copy)</h2>
          <iframe className="report-frame" title="report preview" srcDoc={html} />
        </div>
      )}

      {markdown && (
        <div className="panel">
          <h2>Markdown source</h2>
          <div className="report-md">{markdown}</div>
        </div>
      )}
    </>
  )
}
