import { Page, Status } from '../App'
import './Sidebar.css'

interface Props {
  page: Page
  onNav: (p: Page) => void
  status: Status
}

const statusColor = (s: string) =>
  s === 'running' ? 'green' : s === 'exited' ? 'red' : 'yellow'

export default function Sidebar({ page, onNav, status }: Props) {
  const color = statusColor(status.status)

  const restart = () =>
    fetch('/api/control/restart', { method: 'POST' }).catch(() => {})

  return (
    <nav className="sidebar">
      <div className="sidebar-header">
        <span className="sidebar-logo">Heimdall</span>
        <span className={`badge badge-${color}`}>
          <span className="dot" />
          {status.status}
        </span>
      </div>

      <div className="sidebar-nav">
        {([
          ['cameras',  '📷', 'Cameras'],
          ['settings', '⚙',  'Settings'],
          ['logs',     '📋', 'Logs'],
        ] as [Page, string, string][]).map(([id, icon, label]) => (
          <button
            key={id}
            className={`nav-item ${page === id ? 'active' : ''}`}
            onClick={() => onNav(id)}
          >
            <span>{icon}</span>
            <span>{label}</span>
          </button>
        ))}
      </div>

      <div className="sidebar-footer">
        <button className="btn-danger" style={{ width: '100%' }} onClick={restart}>
          ↺ Restart
        </button>
      </div>
    </nav>
  )
}
