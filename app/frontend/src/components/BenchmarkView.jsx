import { useEffect, useState } from 'react'
import { getJson, postJson } from '../api'

const IMPL_COLORS = { generate: '#1d4ed8', cache: '#15803d', recompute: '#b45309' }

function BarChart({ metric, rows, unit, decimals = 2 }) {
  const max = Math.max(...rows.map((r) => r[metric]), 0.0001)
  return (
    <div className="bar-group">
      <div className="bar-label">{metric.replace(/_/g, ' ')} {unit && `(${unit})`}</div>
      {rows.map((row) => (
        <div className="bar-row" key={row.impl}>
          <span className="impl">{row.impl}</span>
          <div className="bar-track">
            <div
              className={`bar-fill color-${row.impl}`}
              style={{ width: `${Math.max((row[metric] / max) * 100, 1)}%`, background: IMPL_COLORS[row.impl] }}
            />
          </div>
          <span className="bar-val">{Number(row[metric]).toFixed(decimals)}</span>
        </div>
      ))}
    </div>
  )
}

// Meaningful inference comparison across the decoding mechanisms implemented
// in the project (greedy generate / KV-cache / full-recompute), plus the
// recorded GPU speculative-decoding results.
export default function BenchmarkView({ patient, trial, benchmark, setBenchmark }) {
  const [maxTokens, setMaxTokens] = useState(48)
  const [limit, setLimit] = useState(4)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState(null)
  const [spec, setSpec] = useState(null)
  const [specError, setSpecError] = useState(null)

  useEffect(() => {
    getJson('/api/benchmark/speculative')
      .then(setSpec)
      .catch((err) => setSpecError(err.detail || err.message))
  }, [])

  const run = async () => {
    setRunning(true)
    setError(null)
    setBenchmark(null)
    try {
      const result = await postJson('/api/benchmark/decode', {
        patient: { patient_id: patient.patient_id, note: patient.note },
        criteria: trial.criteria.map(({ criterion_type, text }) => ({ criterion_type, text })),
        trial_id: trial.id,
        max_new_tokens: Number(maxTokens),
        criteria_limit: Number(limit),
      })
      setBenchmark(result)
    } catch (err) {
      setError(err.detail || err.message)
    } finally {
      setRunning(false)
    }
  }

  const rows = benchmark ? benchmark.rows.filter((r) => r.ok) : []

  return (
    <>
      <div className="panel">
        <h2>Decode-mechanism comparison (live)</h2>
        <p className="hint">
          The same screening workload is run through the three greedy decoding
          paths implemented in this project: the HF <code>generate</code> path, an
          explicit prefill + incremental <code>KV-cache</code> loop, and a no-cache{' '}
          <code>recompute</code> loop that re-forwards the whole sequence each step.
          Token-stream agreement between paths is verified per criterion - the
          paths are designed to emit identical greedy streams; a divergence would
          indicate a bug (or a numerical knife-edge tie).
        </p>
        <div className="row">
          <label className="hint">
            max_new_tokens
            <input
              type="number" min={8} max={128} value={maxTokens}
              onChange={(e) => setMaxTokens(e.target.value)}
              style={{ width: 90, marginLeft: 6 }}
            />
          </label>
          <label className="hint">
            criteria in workload (capped because recompute is O(n²) per token)
            <input
              type="number" min={1} max={12} value={limit}
              onChange={(e) => setLimit(e.target.value)}
              style={{ width: 70, marginLeft: 6 }}
            />
          </label>
          <button
            className="primary"
            disabled={running || (!patient.note?.trim() && !trial.criteria.length)}
            onClick={run}
          >
            {running ? 'Benchmarking…' : 'Run decode benchmark'}
          </button>
        </div>
        {!patient.note?.trim() && (
          <p className="hint" style={{ color: 'var(--warn)' }}>
            No patient note loaded yet - switch to step 1 first. The benchmark reuses the current patient + criteria context.
          </p>
        )}
        {error && <div className="error-box">{error}</div>}
      </div>

      {benchmark && (
        <div className="panel">
          <h2>Results</h2>
          <p className="hint">{benchmark.meta?.note_about_recompute}</p>
          <table className="simple">
            <thead>
              <tr>
                <th>Decoding mechanism</th>
                <th className="n">Wall (s)</th>
                <th className="n">Generated tokens</th>
                <th className="n">Tokens/s</th>
                <th className="n">Peak RAM delta (MB)</th>
                <th>Per-criterion seconds</th>
              </tr>
            </thead>
            <tbody>
              {benchmark.rows.map((row) => (
                <tr key={row.impl} style={!row.ok ? { color: 'var(--bad)' } : undefined}>
                  <td><code style={{ color: IMPL_COLORS[row.impl] }}>{row.impl}</code></td>
                  <td className="n">{row.wall_seconds}</td>
                  <td className="n">{row.generated_tokens}</td>
                  <td className="n">{row.tokens_per_second}</td>
                  <td className="n">
                    {row.peak_memory_delta_bytes == null
                      ? '-'
                      : (row.peak_memory_delta_bytes / 1048576).toFixed(1)}
                  </td>
                  <td>{row.per_criterion_seconds.join(', ')}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {rows.length > 0 && (
            <div className="bars">
              <BarChart metric="wall_seconds" rows={rows} unit="s" />
              <BarChart metric="tokens_per_second" rows={rows} unit="tok/s" />
            </div>
          )}

          {(benchmark.rows.some((r) => !r.ok)) && (
            <div className="error-box">
              Some paths failed:{' '}
              {benchmark.rows.filter((r) => !r.ok).map((r) => `${r.impl}: ${r.error}`).join(' | ')}
            </div>
          )}

          <h3>Token-stream agreement per criterion</h3>
          <table className="simple">
            <thead>
              <tr>
                <th>#</th>
                <th>Criterion</th>
                <th>All paths identical</th>
                <th>First divergence (per impl)</th>
                <th>Labels</th>
              </tr>
            </thead>
            <tbody>
              {benchmark.agreement.map((entry) => (
                <tr key={entry.index}>
                  <td className="n">{entry.index}</td>
                  <td>{entry.criterion_text}</td>
                  <td>{entry.all_equal ? '✓ identical' : '✗ divergent'}</td>
                  <td>
                    {entry.first_divergence &&
                      Object.entries(entry.first_divergence)
                        .map(([impl, pos]) => `${impl}: ${pos ?? '-'}`)
                        .join(', ')}
                  </td>
                  <td>{entry.labels_equal ? 'same' : entry.labels.generate + ' vs ' + entry.labels.cache}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="panel">
        <h2>
          Speculative decoding (greedy, draft E2B -> target E4B)
          {spec && <span className="recorded-tag">recorded A100 run - not live</span>}
        </h2>
        <p className="hint">
          The project also implements greedy speculative decoding with KV caches
          (draft proposes γ tokens, one batched target forward verifies the block).
          It requires the GPU-only E4B/E2B checkpoint pair, so the app displays the
          recorded research benchmark from the A100 host instead of running it live.
        </p>
        {specError && (
          <div className="error-box">{specError}</div>
        )}
        {spec && (
          <>
            <p className="hint">
              {spec.environment.gpu_name} · torch {spec.environment.torch} ·{' '}
              bf16 · target {spec.environment.target_model_id} · draft{' '}
              {spec.environment.draft_model_id}
            </p>
            <table className="simple">
              <thead>
                <tr>
                  <th className="n">γ</th>
                  <th className="n">Acceptance rate α</th>
                  <th className="n">Tokens per target forward</th>
                  <th className="n">Median wall (s)</th>
                  <th className="n">Baseline greedy (s)</th>
                  <th className="n">Speedup vs baseline</th>
                </tr>
              </thead>
              <tbody>
                {spec.gamma_rows.map((row) => (
                  <tr key={row.gamma}>
                    <td className="n">{row.gamma}</td>
                    <td className="n">{row.acceptance_rate?.toFixed(3)}</td>
                    <td className="n">{row.mean_tokens_per_target_forward?.toFixed(2)}</td>
                    <td className="n">{row.median_wall_seconds}</td>
                    <td className="n">{row.baseline_wall_seconds}</td>
                    <td className="n">
                      {row.speedup_vs_baseline
                        ? `${row.speedup_vs_baseline}× ${row.speedup_vs_baseline < 1 ? '(slowdown)' : '(faster)'}`
                        : '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="hint">
              Cost model from the recorded run: one draft step costs ≈
              {spec.step_time_summary.draft_share_percent
                ? `${spec.step_time_summary.draft_share_percent?.toFixed(1)}% of a target step`
                : '-'}
              , so speculative decoding only wins when α exceeds the draft share;
              here it never does (α tops out ≈0.69 at γ=2). 0.33–0.59× across cells,
              a correctly-measured slowdown for this checkpoint pair.
            </p>
            <div className="spec-note">
              <strong>Correctness:</strong> {spec.agreement.summary}{' '}
              {spec.agreement.token_exact_runs}/{spec.agreement.n_prompts} prompt runs were
              token-exact at every γ; the remainder diverge exactly once at a bf16
              knife-edge tie (top-2 logit margin 0.0).
            </div>
          </>
        )}
      </div>
    </>
  )
}
