import { useEffect, useState } from 'react'
import './Page.css'

export default function SettingsPage() {
  const [settings, setSettings] = useState<any>(null)
  const [draft, setDraft] = useState<any>(null)
  const [saving, setSaving] = useState(false)
  const [msg, setMsg] = useState('')

  useEffect(() => {
    fetch('/api/settings')
      .then(r => r.json())
      .then(d => { setSettings(d); setDraft(d) })
  }, [])

  if (!draft) return <div className="page-loading">Loading…</div>

  const setNested = (path: string[], value: any) => {
    setDraft((prev: any) => {
      const next = structuredClone(prev)
      let node = next
      for (let i = 0; i < path.length - 1; i++) node = node[path[i]]
      node[path[path.length - 1]] = value
      return next
    })
  }

  const save = async () => {
    setSaving(true)
    const res = await fetch('/api/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(draft),
    })
    setSaving(false)
    setMsg(res.ok ? 'Saved. Restart required.' : 'Save failed.')
  }

  const t = draft.tracker ?? {}
  const c = draft.comm ?? {}

  return (
    <div>
      <div className="page-header"><h1>Settings</h1></div>

      <div className="settings-card">
        <div className="settings-section-title">Tracker</div>

        <div className="settings-grid">
          {[
            ['confirmation_frames', 'Confirmation Frames', 0, 20, 1],
            ['loss_frames',         'Loss Frames',         0, 60, 1],
            ['gate_distance',       'Gate Distance (m)',   0, 10, 0.1],
            ['clutter_density',     'Clutter Density',     0, 10, 0.1],
            ['p_detection',         'P(Detection)',         0, 1,  0.01],
            ['meas_noise_r',        'Meas Noise R',        0, 2,  0.01],
            ['process_noise_q',     'Process Noise Q',     0, 10, 0.1],
            ['pos_cov_floor',       'Pos Cov Floor',       0, 2,  0.01],
          ].map(([key, label, min, max, step]) => (
            <div className="settings-field" key={key as string}>
              <label>{label as string}</label>
              <div className="settings-num-row">
                <input type="range" min={min} max={max} step={step}
                  value={t[key as string] ?? 0}
                  onChange={e => setNested(['tracker', key as string], Number(e.target.value))} />
                <input type="number" min={min} max={max} step={step}
                  style={{ width: 80 }}
                  value={t[key as string] ?? 0}
                  onChange={e => setNested(['tracker', key as string], Number(e.target.value))} />
              </div>
            </div>
          ))}
        </div>

        <hr />
        <div className="settings-section-title">Filter Model</div>
        <div className="settings-field">
          <label>Motion Model</label>
          <select value={t.filter_model ?? 'constant_velocity'}
            onChange={e => setNested(['tracker', 'filter_model'], e.target.value)}>
            <option value="constant_position">Constant Position</option>
            <option value="constant_velocity">Constant Velocity</option>
            <option value="constant_acceleration">Constant Acceleration</option>
          </select>
        </div>

        <hr />
        <div className="settings-section-title">ZeroMQ Comm</div>
        <div className="settings-grid">
          {[
            ['pose_bind_addr',         'Pose Bind (in)'],
            ['output_bind_addr',       'Output Bind (out)'],
            ['apriltag_pose_bind_addr','AprilTag Pose Bind'],
          ].map(([key, label]) => (
            <div className="settings-field" key={key}>
              <label>{label}</label>
              <input type="text" value={c[key] ?? ''}
                onChange={e => setNested(['comm', key], e.target.value)} />
            </div>
          ))}
        </div>

        <hr />
        <div className="settings-section-title">Misc</div>
        <div className="settings-field">
          <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <input type="checkbox" checked={!!draft.bypass_tracker}
              onChange={e => setNested(['bypass_tracker'], e.target.checked)} />
            Bypass Tracker (raw detections)
          </label>
        </div>

        <div className="card-actions" style={{ marginTop: 20 }}>
          {msg && <span style={{ color: 'var(--muted)', fontSize: 12, flex: 1 }}>{msg}</span>}
          <button className="btn-ghost"
            onClick={() => { setDraft(structuredClone(settings)); setMsg('') }}>
            Revert
          </button>
          <button className="btn-primary" onClick={save} disabled={saving}>
            {saving ? 'Saving…' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  )
}
