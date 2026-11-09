import { useState } from 'react'

const toCriteria = (source) =>
  source.criteria.map((c) => ({
    criterion_type: c.criterion_type,
    text: c.text,
    gold_label: c.gold_label,
    gold_evidence_ids: c.gold_evidence_ids,
  }))

// Step 2: trial criteria. Pick a bundled trial, edit any criterion, add or
// delete rows, or start from an empty free-form set. Each criterion is scored
// independently by the model.
export default function TrialPicker({ library, trial, setTrial, results, setResults, onNext }) {
  const [customTitle, setCustomTitle] = useState(trial.origin === 'custom' ? trial.title : '')

  const selectBundled = (trialId) => {
    const source = library.trials.find((t) => t.id === trialId)
    if (!source) return
    setTrial({
      id: source.id,
      title: source.title,
      criteria: toCriteria(source),
      origin: 'library',
    })
    if (results) setResults(null)
  }

  const startFreeForm = () => {
    setTrial({ id: null, title: '', criteria: [], origin: 'custom' })
    setCustomTitle('')
    if (results) setResults(null)
  }

  const addBlank = (type) => {
    setTrial({
      ...trial,
      criteria: [...trial.criteria, { criterion_type: type, text: '' }],
    })
    if (results) setResults(null)
  }

  const update = (index, patch) => {
    const next = trial.criteria.map((c, i) => (i === index ? { ...c, ...patch } : c))
    setTrial({ ...trial, criteria: next })
    if (results) setResults(null)
  }

  const remove = (index) => {
    setTrial({ ...trial, criteria: trial.criteria.filter((_, i) => i !== index) })
    if (results) setResults(null)
  }

  const setTitle = (title) => {
    setCustomTitle(title)
    setTrial(prev => ({ ...prev, title }))
  }

  const nInclusion = trial.criteria.filter((c) => c.criterion_type === 'inclusion').length
  const nExclusion = trial.criteria.filter((c) => c.criterion_type === 'exclusion').length

  return (
    <>
      <div className="panel">
        <h2>Choose a bundled trial</h2>
        <p className="hint">
          Bundled sets come from the TrialGPT-Criterion-Annotations test split. Selecting one
          replaces the current criterion list below.
        </p>
        {library && (
          <div className="library-list">
            {library.trials.map((t) => (
              <div
                key={t.id}
                className={`trial-tile ${trial.id === t.id ? 'selected' : ''}`}
                onClick={() => selectBundled(t.id)}
              >
                <div className="title">{t.title}</div>
                <div className="sub">{t.n_criteria} criteria</div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="panel">
        <div className="row spread">
          <h2 style={{ marginBottom: 0 }}>Criterion set ({trial.criteria.length})</h2>
          <button className="secondary small" onClick={startFreeForm}>Start free-form (empty)</button>
        </div>
        <p className="hint">
          {nInclusion} inclusion, {nExclusion} exclusion. Edit any row, delete, or add
          criteria. Free-form criteria are screened with the same model prompt as the
          bundled ones.
        </p>

        {trial.origin === 'custom' && (
          <input
            type="text"
            placeholder="Trial title (optional, e.g. 'My hypothetical sepsis trial')"
            value={customTitle}
            onChange={(e) => setTitle(e.target.value)}
            style={{ marginBottom: 12, maxWidth: 480 }}
          />
        )}

        {trial.criteria.map((criterion, index) => (
          <div className="criterion-edit-row" key={`${index}-${criterion.text}`}>
            <select
              value={criterion.criterion_type}
              onChange={(e) => update(index, { criterion_type: e.target.value })}
              aria-label={`criterion ${index + 1} type`}
            >
              <option value="inclusion">inclusion</option>
              <option value="exclusion">exclusion</option>
            </select>
            <input
              type="text"
              placeholder="Criterion text, e.g. 'Age greater than 18 years'"
              value={criterion.text}
              onChange={(e) => update(index, { text: e.target.value })}
            />
            <button
              className="small ghost danger"
              title="Remove criterion"
              onClick={() => remove(index)}
            >
              ✕
            </button>
          </div>
        ))}

        <div className="row" style={{ marginTop: 8 }}>
          <button className="secondary small" onClick={() => addBlank('inclusion')}>
            + inclusion criterion
          </button>
          <button className="secondary small" onClick={() => addBlank('exclusion')}>
            + exclusion criterion
          </button>
          <span style={{ flex: 1 }} />
          <button
            className="primary"
            disabled={!trial.criteria.length || trial.criteria.some((c) => !c.text.trim())}
            onClick={onNext}
          >
            Next: run screening →
          </button>
        </div>
      </div>
    </>
  )
}
