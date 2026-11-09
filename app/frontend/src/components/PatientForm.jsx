import { useState } from 'react'
import { postJson } from '../api'

// Step 1: patient record. Users paste a free-text (or pre-numbered) note; the
// backend normalizes it into the exact 'N. sentence' format the model expects.
export default function PatientForm({
  library,
  patient,
  setPatient,
  notePreview,
  setNotePreview,
  setTrial,
  results,
  setResults,
  onNext,
}) {
  const [normalizing, setNormalizing] = useState(false)
  const [error, setError] = useState(null)

  const loadSample = (trialId) => {
    if (!library) return
    const source = library.trials.find((t) => t.id === trialId)
    if (!source) return
    // Select the pair sample + bundled criteria set together (they co-vary in
    // the source dataset: one patient record per trial bundle). Gold fields are
    // kept for the model-vs-expert comparison step.
    setPatient({ patient_id: source.id, note: source.note })
    setTrial({
      id: source.id,
      title: source.title,
      criteria: source.criteria.map((c) => ({
        criterion_type: c.criterion_type,
        text: c.text,
        gold_label: c.gold_label,
        gold_evidence_ids: c.gold_evidence_ids,
      })),
      origin: 'library',
    })
    setNotePreview(null)
    if (results) setResults(null)
  }

  const normalize = async () => {
    setNormalizing(true)
    setError(null)
    try {
      const preview = await postJson('/api/normalize-note', { note: patient.note })
      setNotePreview({ note: preview.note, sentences: preview.sentences })
    } catch (err) {
      setError(err.detail || err.message)
    } finally {
      setNormalizing(false)
    }
  }

  const nCriteria = trial.criteria.length

  return (
    <>
      <div className="panel">
        <h2>Patient record</h2>
        <p className="hint">
          Paste a clinical note as free text — it will be split into sentences and
          auto-numbered (0, 1, 2…) exactly the way the model was trained. Sentences
          the model later cites are pointed at this numbering, so keep it stable.
        </p>

        <label className="hint" htmlFor="patient-id">Record identifier (optional)</label>
        <input
          id="patient-id"
          type="text"
          placeholder="e.g. demo-patient-01"
          value={patient.patient_id}
          onChange={(e) => setPatient({ ...patient, patient_id: e.target.value })}
          style={{ maxWidth: 320, marginBottom: 12 }}
        />

        <label className="hint" htmlFor="note">
          Patient note ({patient.note.length} characters)
        </label>
        <textarea
          id="note"
          rows={10}
          placeholder={'Example:\nThe patient is a 62-year-old woman with type 2 diabetes.\nHbA1c measured last week was 8.9%.\nShe reports adequate access to transportation.'}
          value={patient.note}
          onChange={(e) => {
            setPatient({ ...patient, note: e.target.value })
            if (notePreview) setNotePreview(null) // stale numbering after an edit
          }}
        />
        {error && <div className="error-box">Note rejected: {error}</div>}

        <div className="row spread" style={{ marginTop: 10 }}>
          <button className="primary" disabled={!patient.note.trim() || normalizing} onClick={normalize}>
            {normalizing ? 'Normalizing…' : 'Normalize & preview sentences'}
          </button>
          <button className="primary" disabled={!notePreview} onClick={onNext}>
            Next: trial criteria →
          </button>
        </div>
      </div>

      {notePreview && (
        <div className="panel">
          <h2>Numbered sentences ({notePreview.sentences.length})</h2>
          <p className="hint">
            This exact numbering enters the prompt. Evidence in results refers to
            these IDs. The note also carries into later steps.
          </p>
          <div className="note-preview">{notePreview.note}</div>
        </div>
      )}

      <div className="panel">
        <h2>Or load a bundled sample record</h2>
        <p className="hint">
          Sample records come with their trial's criterion set pre-selected and
          reference (expert) annotations, so model vs. expert can be compared later.
          Loading a sample replaces the note {nCriteria > 0 ? 'and current criterion set ' : ''}
          above.
        </p>
        {library && (
          <div className="library-list">
            {library.trials.map((t) => (
              <div
                key={t.id}
                className={`trial-tile ${patient.patient_id === t.id ? 'selected' : ''}`}
                onClick={() => loadSample(t.id)}
              >
                <div className="title">{t.title}</div>
                <div className="sub">
                  {t.n_criteria} criteria · {t.n_note_sentences} note sentences
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  )
}
