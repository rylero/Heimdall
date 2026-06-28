import { useState, useRef, useEffect } from 'react'
import './CameraCard.css'

interface Props {
  camera: any
  onSaved: () => void
}

// Known V4L2 controls to show as sliders
const LIVE_CONTROLS = [
  { key: 'exposure_absolute',    label: 'Exposure',   min: 1,   max: 5000 },
  { key: 'gain',                 label: 'Gain',       min: 0,   max: 255  },
  { key: 'brightness',           label: 'Brightness', min: -64, max: 64   },
  { key: 'contrast',             label: 'Contrast',   min: 0,   max: 95   },
]

function VideoPreview({ device, name }: { device?: string; name: string }) {
  // MediaMTX serves HLS at /name/index.m3u8 on port 8888
  // For apriltag stream specifically
  const streamName = name === 'apriltag' ? 'apriltag' : 'ds-test'
  const hlsUrl = `http://${window.location.hostname}:8888/${streamName}/index.m3u8`

  return (
    <div className="video-preview">
      <video
        autoPlay muted playsInline
        style={{ width: '100%', borderRadius: 6, background: '#000', maxHeight: 200 }}
        onError={e => { (e.target as HTMLVideoElement).style.display = 'none' }}
        ref={el => {
          if (!el) return
          if ((window as any).Hls) {
            const hls = new (window as any).Hls()
            hls.loadSource(hlsUrl)
            hls.attachMedia(el)
          } else if (el.canPlayType('application/vnd.apple.mpegurl')) {
            el.src = hlsUrl
          }
        }}
      />
      <div className="video-label">{hlsUrl}</div>
    </div>
  )
}

export default function CameraCard({ camera, onSaved }: Props) {
  const [open, setOpen] = useState(false)
  const [draft, setDraft] = useState<any>(() => structuredClone(camera))
  const [liveCtrl, setLiveCtrl] = useState<Record<string, number>>({})
  const [saving, setSaving] = useState(false)
  const [msg, setMsg] = useState('')
  const name = camera._name

  useEffect(() => { setDraft(structuredClone(camera)) }, [camera])

  const set = (path: string[], value: any) => {
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
    try {
      const res = await fetch(`/api/cameras/${name}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(draft),
      })
      if (res.ok) { setMsg('Saved. Restart required for pipeline changes.'); onSaved() }
      else setMsg('Save failed.')
    } finally { setSaving(false) }
  }

  const applyV4l2 = async (ctrl: string, value: number) => {
    await fetch(`/api/cameras/${name}/v4l2`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ [ctrl]: value }),
    })
  }

  const isApriltag = camera._is_apriltag

  return (
    <div className="camera-card">
      <div className="camera-card-header" onClick={() => setOpen(o => !o)}>
        <div className="camera-card-title">
          <span className="chevron">{open ? '▼' : '▶'}</span>
          <span className="cam-name">{name}</span>
          {draft.device && <span className="cam-device">{draft.device}</span>}
        </div>
        <span className={`badge ${isApriltag ? 'badge-yellow' : 'badge-green'}`}>
          <span className="dot" />
          {isApriltag ? 'AprilTag' : 'Detection'}
        </span>
      </div>

      {open && (
        <div className="camera-card-body">
          {/* Live preview */}
          <VideoPreview device={draft.device} name={name} />

          {/* Live V4L2 controls */}
          {!isApriltag && (
            <>
              <div className="section-label">Live V4L2 Controls</div>
              <div className="live-controls">
                {LIVE_CONTROLS.map(ctrl => (
                  <div key={ctrl.key} className="ctrl-row">
                    <label>{ctrl.label}</label>
                    <div className="ctrl-slider-row">
                      <input
                        type="range"
                        min={ctrl.min} max={ctrl.max}
                        value={liveCtrl[ctrl.key] ?? Math.floor((ctrl.min + ctrl.max) / 2)}
                        onChange={e => {
                          const v = Number(e.target.value)
                          setLiveCtrl(p => ({ ...p, [ctrl.key]: v }))
                        }}
                        onMouseUp={e => applyV4l2(ctrl.key, Number((e.target as HTMLInputElement).value))}
                      />
                      <span className="ctrl-value">
                        {liveCtrl[ctrl.key] ?? '—'}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
              <hr />
            </>
          )}

          {/* Pipeline config */}
          <div className="section-label">
            Pipeline Config <span className="restart-badge">⚠ requires restart</span>
          </div>
          <div className="form-grid">
            <div className="form-field">
              <label>Device</label>
              <input type="text" value={draft.device ?? ''} onChange={e => set(['device'], e.target.value)} />
            </div>
            {!isApriltag && (
              <>
                <div className="form-field-row">
                  <div className="form-field">
                    <label>Width</label>
                    <input type="number" value={draft.width ?? ''} onChange={e => set(['width'], Number(e.target.value))} />
                  </div>
                  <div className="form-field">
                    <label>Height</label>
                    <input type="number" value={draft.height ?? ''} onChange={e => set(['height'], Number(e.target.value))} />
                  </div>
                  <div className="form-field">
                    <label>FPS</label>
                    <input type="number" value={draft.fps ?? ''} onChange={e => set(['fps'], Number(e.target.value))} />
                  </div>
                </div>
                <div className="form-field-row checkboxes">
                  <label className="checkbox-label">
                    <input type="checkbox" checked={!!draft.flip_h} onChange={e => set(['flip_h'], e.target.checked)} />
                    Flip H
                  </label>
                  <label className="checkbox-label">
                    <input type="checkbox" checked={!!draft.flip_v} onChange={e => set(['flip_v'], e.target.checked)} />
                    Flip V
                  </label>
                  <label className="checkbox-label">
                    <input type="checkbox" checked={!!draft.hw_decode} onChange={e => set(['hw_decode'], e.target.checked)} />
                    HW Decode
                  </label>
                </div>
              </>
            )}
            {isApriltag && (
              <div className="form-field-row">
                <div className="form-field">
                  <label>Width</label>
                  <input type="number" value={draft.width ?? ''} onChange={e => set(['width'], Number(e.target.value))} />
                </div>
                <div className="form-field">
                  <label>Height</label>
                  <input type="number" value={draft.height ?? ''} onChange={e => set(['height'], Number(e.target.value))} />
                </div>
                <div className="form-field">
                  <label>FPS</label>
                  <input type="number" value={draft.fps ?? ''} onChange={e => set(['fps'], Number(e.target.value))} />
                </div>
              </div>
            )}
          </div>

          {/* Intrinsics */}
          {draft.intrinsics && (
            <>
              <div className="section-label" style={{ marginTop: 12 }}>Intrinsics (calibration)</div>
              <div className="form-grid intrinsics">
                {['fx','fy','cx','cy'].map(k => (
                  <div className="form-field" key={k}>
                    <label>{k}</label>
                    <input type="number" step="0.001"
                      value={draft.intrinsics[k] ?? ''}
                      onChange={e => set(['intrinsics', k], Number(e.target.value))} />
                  </div>
                ))}
                {draft.intrinsics.distortion && ['k1','k2','p1','p2','k3'].map(k => (
                  <div className="form-field" key={k}>
                    <label>{k}</label>
                    <input type="number" step="0.000001"
                      value={draft.intrinsics.distortion[k] ?? ''}
                      onChange={e => set(['intrinsics', 'distortion', k], Number(e.target.value))} />
                  </div>
                ))}
              </div>
            </>
          )}

          <div className="card-actions">
            {msg && <span className="save-msg">{msg}</span>}
            <button className="btn-ghost" onClick={() => { setDraft(structuredClone(camera)); setMsg('') }}>
              Revert
            </button>
            <button className="btn-primary" onClick={save} disabled={saving}>
              {saving ? 'Saving…' : 'Save'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
