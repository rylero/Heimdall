import { useState, useEffect } from 'react'
import Sidebar from './components/Sidebar'
import CamerasPage from './pages/CamerasPage'
import SettingsPage from './pages/SettingsPage'
import LogsPage from './pages/LogsPage'
import './App.css'

export type Page = 'cameras' | 'settings' | 'logs'

export interface Status { status: string }

export default function App() {
  const [page, setPage] = useState<Page>('cameras')
  const [status, setStatus] = useState<Status>({ status: 'unknown' })

  useEffect(() => {
    const poll = () =>
      fetch('/api/control/status')
        .then(r => r.json())
        .then(setStatus)
        .catch(() => setStatus({ status: 'error' }))
    poll()
    const id = setInterval(poll, 3000)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="app">
      <Sidebar page={page} onNav={setPage} status={status} />
      <main className="content">
        {page === 'cameras'  && <CamerasPage />}
        {page === 'settings' && <SettingsPage />}
        {page === 'logs'     && <LogsPage />}
      </main>
    </div>
  )
}
