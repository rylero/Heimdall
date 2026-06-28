import { useEffect, useState } from 'react'
import CameraCard from '../components/CameraCard'
import './Page.css'

export default function CamerasPage() {
  const [cameras, setCameras] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  const load = () =>
    fetch('/api/cameras')
      .then(r => r.json())
      .then(d => { setCameras(d); setLoading(false) })
      .catch(() => setLoading(false))

  useEffect(() => { load() }, [])

  if (loading) return <div className="page-loading">Loading cameras…</div>

  return (
    <div>
      <div className="page-header">
        <h1>Cameras</h1>
      </div>
      <div className="card-stack">
        {cameras.map(cam => (
          <CameraCard key={cam._name} camera={cam} onSaved={load} />
        ))}
      </div>
    </div>
  )
}
