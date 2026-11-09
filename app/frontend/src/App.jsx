import { useEffect, useMemo, useState } from 'react'
import { getJson } from './api'
import DisclaimerBanner from './components/Disclaimer'
import BackendStatus from './components/BackendStatus'
import PatientForm from './components/PatientForm'
import TrialPicker from './components/TrialPicker'
import ResultsView from './components/ResultsView'
import BenchmarkView from './components/BenchmarkView'
import ReportExport from './components/ReportExport'

export default function App() {
  const [engine, setEngine] = useState(null)
  const [backendReady, setBackendReady] = useState(false)
  const [library, setLibrary] = useState(null)
  const [loadError, setLoadError] = useState(null)
  const [tab, setTab] = useState('patient')

  // Shared application state (one screening session at a time).
  const [patient, setPatient] = useState({ patient_id: '', note: '' })
  const [notePreview, setNotePreview] = useState(null)
  const [trial, setTrial] = useState({ id: null, title: '', criteria: [], origin: 'custom' })
  const [results, setResults] = useState(null) // {meta, results} built progressively
  const [running, setRunning] = useState(false)
  const [benchmark, setBenchmark] = useState(null)

  const hasNote = Boolean(patient.note.trim())
  const hasCriteria = trial.criteria.length > 0 && trial.criteria.every((c) => c.text.trim())
  const hasResults = Boolean(results && results.results.length > 0)

  // Only fetch the library once the backend is confirmed reachable; the
  // BackendStatus component keeps probing (with timeouts) until it is.
  useEffect(() => {
    if (!backendReady) return
    getJson('/api/library')
      .then(setLibrary)
      .catch((err) => setLoadError(err.detail || err.message))
  }, [backendReady])

  const engineTag = useMemo(() => {
    if (!engine) return 'connecting to backend…'
    const device = engine.device === 'cuda' ? 'GPU (cuda)' : 'CPU'
    const suffix = engine.adapter_overridden || engine.base_model_overridden ? ' [env override]' : ''
    return `${engine.base_model_id} · ${device}${suffix}`
  }, [engine])

  const tabs = [
    { id: 'patient', label: '1 · Patient record', enabled: true },
    { id: 'trial', label: '2 · Trial criteria', enabled: hasNote },
    { id: 'results', label: '3 · Screening results', enabled: hasNote && hasCriteria },
    { id: 'export', label: '4 · Export report', enabled: hasResults },
    { id: 'benchmark', label: 'Decode benchmark', enabled: true },
  ]

  return (
    <div className="app">
      <DisclaimerBanner />
      <BackendStatus
        onReady={(info) => {
          setEngine(info)
          setBackendReady(true)
        }}
      />
      <header className="main-header">
        <h1>TrialO - Eligibility Screening</h1>
        <span className="engine-tag">
          Model: {engineTag} · Educational prototype, not a clinical enrollment decision.
        </span>
      </header>

      <nav className="tabs">
        {tabs.map((t) => (
          <button
            key={t.id}
            className={tab === t.id ? 'active' : ''}
            disabled={!t.enabled}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      {loadError && <div className="error-box">Backend unavailable: {loadError}</div>}

      {tab === 'patient' && (
        <PatientForm
          library={library}
          patient={patient}
          setPatient={setPatient}
          trial={trial}
          notePreview={notePreview}
          setNotePreview={setNotePreview}
          setTrial={setTrial}
          results={results}
          setResults={setResults}
          onNext={() => setTab('trial')}
        />
      )}

      {tab === 'trial' && (
        <TrialPicker
          library={library}
          trial={trial}
          setTrial={setTrial}
          results={results}
          setResults={setResults}
          onNext={() => setTab('results')}
        />
      )}

      {tab === 'results' && (
        <ResultsView
          patient={patient}
          notePreview={notePreview}
          trial={trial}
          results={results}
          setResults={setResults}
          running={running}
          setRunning={setRunning}
          setNotePreview={setNotePreview}
          hasResults={hasResults}
          onExport={() => setTab('export')}
          onBenchmark={() => setTab('benchmark')}
        />
      )}

      {tab === 'export' && (
        <ReportExport
          patient={patient}
          trial={trial}
          results={results}
          benchmark={benchmark}
        />
      )}

      {tab === 'benchmark' && (
        <BenchmarkView
          patient={patient}
          trial={trial}
          benchmark={benchmark}
          setBenchmark={setBenchmark}
        />
      )}

      <div className="footer">
        Backend API documentation: <code>/api/docs</code> (served by the backend, port 8000).
        <br />
        Sentence evidence is quoted verbatim from the patient note you submit; nothing leaves
        your machine except requests to the locally hosted backend and the one-time base-model
        download from the Hugging Face Hub.
      </div>
    </div>
  )
}
