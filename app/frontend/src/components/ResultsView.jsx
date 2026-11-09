import { useEffect, useRef, useState } from 'react'
import { postJson, postSse } from '../api'

const LABEL_CLASS = {
  included: 'chip included',
  'not included': 'chip excluded',
  excluded: 'chip excluded',
  'not excluded': 'chip included',
  'not applicable': 'chip not',
  'not enough information': 'chip missing',
}

function chipClass(label, valid) {
  if (!valid || !label) return 'chip error'
  return LABEL_CLASS[label] || 'chip not'
}

// Step 3: run the Transformer and inspect every criterion-level result with
// exact supporting evidence sentences quoted from the note.
export default function ResultsView({
  patient,
  notePreview,
  trial,
  results,
  setResults,
  running,
  setRunning,
  setNotePreview,
  hasResults,
  onExport,
  onBenchmark,
}) {
  const [error, setError] = useState(null)
  const [info, setInfo] = useState('')
  const cancelRef = useRef(false)

  // Keep the normalized preview current with whatever note is loaded.
  useEffect(() => {
    if (!patient.note || notePreview) return
    let cancelled = false
    postJson('/api/normalize-note', { note: patient.note })
      .then((preview) => {
        if (!cancelled) {
          setNotePreview({ note: preview.note, sentences: preview.sentences })
        }
      })
      .catch((err) => setInfo(`Normalization failed: ${err.detail || err.message}`))
    return () => {
      cancelled = true
    }
  }, [patient.note]) // eslint-disable-line react-hooks/exhaustive-deps

  const sentences = notePreview ? notePreview.sentences : []

  const run = async () => {
    setRunning(true)
    cancelRef.current = false
    setError(null)
    setResults({ meta: null, results: [], done: false })

    try {
      await postSse('/api/screen/stream', {
        patient: { patient_id: patient.patient_id, note: patient.note },
        criteria: trial.criteria.map(({ criterion_type, text }) => ({ criterion_type, text })),
        trial_id: trial.id,
        trial_title: trial.title || null,
        decoding_impl: 'generate',
      }, (event) => {
        if (cancelRef.current) return
        if (event.type === 'meta') {
          setResults((prev) => ({ ...prev, meta: event.meta }))
        } else if (event.type === 'result') {
          setResults((prev) => ({ ...prev, results: [...prev.results, event.result] }))
        } else if (event.type === 'done') {
          setResults((prev) => ({ ...prev, done: true }))
        }
      })
    } catch (err) {
      setError(err.detail || err.message)
    } finally {
      setRunning(false)
    }
  }

  const stop = () => {
    cancelRef.current = true
    setRunning(false)
  }

  const counts = {}
  for (const r of results ? results.results : []) {
    const key = r.valid && r.label ? r.label : 'unparsable'
    counts[key] = (counts[key] || 0) + 1
  }

  const evidenceIds = new Set(
    (results ? results.results : []).flatMap((r) => r.evidence.map((e) => e.sentence_id)),
  )

  const renderNote = () => {
    if (!notePreview) return null
    return notePreview.sentences.map((sentence, i) => (
      <div key={i} className={evidenceIds.has(i) ? 'evidence' : ''}>
        <strong style={{ userSelect: 'none' }}>{i}. </strong>
        {sentence}
      </div>
    ))
  }

  return (
    <>
      <div className="panel">
        <div className="row spread">
          <div>
            <h2>Run screening</h2>
            <p className="hint" style={{ marginBottom: 0 }}>
              {trial.criteria.length} criteria · note with {sentences.length || '…'} sentences ·
              decoding: <code>generate</code> (greedy)
              {trial.title ? '' : ''}
            </p>
          </div>
          <div className="row">
            {running ? (
              <button className="secondary" onClick={stop}>Stop</button>
            ) : null}
            <button className="primary" disabled={running} onClick={run}>
              {running ? 'Screening…' : 'Run screening'}
            </button>
          </div>
        </div>

        {running && (
          <div className="progress">
            <div
              style={{
                width: results && results.results.length
                  ? `${(results.results.length / trial.criteria.length) * 100}%`
                  : '4%',
              }}
            />
          </div>
        )}
        {info && <p className="hint">{info}</p>}
        {error && <div className="error-box">{error}</div>}
        {results && (
          <p className="hint" style={{ marginTop: 8 }}>
            {results.results.length}/{trial.criteria.length} scored
            {Object.keys(counts).length > 0
              ? ' · ' + Object.entries(counts).map(([label, n]) => `${n} ${label}`).join(', ')
              : ''}
          </p>
        )}
      </div>

      {results && results.results.map((r, i) => (
        <div className="criterion-card" key={i}>
          <div className="head">
            <div>
              <div className="crit-text">
                {i + 1}. <span className="chip not" style={{ marginRight: 6 }}>{r.criterion_type}</span>
                {r.criterion_text}
              </div>
              <div className="meta">
                {r.prompt_tokens} prompt tokens · {r.generated_tokens} generated ·
                {' '}{r.wall_seconds}s · stopped_on_eos {String(r.stopped_on_eos)}
              </div>
            </div>
            <span className={chipClass(r.label, r.valid)}>{r.label || 'unparsable'}</span>
          </div>

          {r.evidence.length > 0 ? (
            <ol className="evidence">
              {r.evidence.map((e) => (
                <li key={e.sentence_id}>
                  <strong>[{e.sentence_id}]</strong> "{e.text}"
                </li>
              ))}
            </ol>
          ) : (
            <p className="meta" style={{ margin: '6px 0' }}>No supporting evidence cited.</p>
          )}

          {r.explanation && <p className="explanation">{r.explanation}</p>}
          {(r.warnings.length > 0 || r.errors.length > 0) && (
            <p className="meta" style={{ color: 'var(--warn)' }}>
              {r.errors.length > 0 && <>Errors: {r.errors.join('; ')}</>}
              {r.errors.length > 0 && r.warnings.length > 0 && ' · '}
              {r.warnings.length > 0 && <>Warnings: {r.warnings.join('; ')}</>}
            </p>
          )}

          <details className="raw">
            <summary>Model raw output</summary>
            <pre>{r.raw_text || '(empty)'}</pre>
          </details>

          {trial.origin === 'library' && (() => {
            const bundled = trial.criteria[i] && trial.criteria[i].gold_label != null ? true : false
            if (!bundled) return null
            const gold = trial.criteria[i]
            const agrees = gold.gold_label === r.label
            return (
              <div className="gold">
                Reference (expert) annotation: <strong>{gold.gold_label}</strong>{' '}
                {agrees ? '✓ agrees with model' : `✗ differs from model`}
                {gold.gold_evidence_ids ? gold.gold_evidence_ids.length
                  ? ` · evidence ids [${gold.gold_evidence_ids.join(', ')}]`
                  : ' · no expert-cited evidence' : ''}
              </div>
            )
          })()}
        </div>
      ))}

      <div className="panel">
        <h2>Patient note (cited sentences highlighted)</h2>
        {!notePreview && <p className="hint">Loading…</p>}
        {notePreview && <div className="note-preview" style={{ maxHeight: 340 }}>{renderNote()}</div>}
      </div>

      {hasResults && (
        <div className="row">
          <button className="primary" onClick={onExport}>Export report -></button>
          <button className="secondary" onClick={onBenchmark}>Benchmark this run's decoding -></button>
        </div>
      )}
    </>
  )
}
