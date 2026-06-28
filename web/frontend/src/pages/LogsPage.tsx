import { useEffect, useRef, useState } from 'react'
import './Page.css'

type Tab = 'container' | 'tracker'

export default function LogsPage() {
  const [tab, setTab] = useState<Tab>('container')
  const [lines, setLines] = useState<string[]>([])
  const [paused, setPaused] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const esRef = useRef<EventSource | null>(null)
  const pausedRef = useRef(false)
  pausedRef.current = paused

  useEffect(() => {
    setLines([])
    esRef.current?.close()
    const es = new EventSource(`/api/logs/${tab}`)
    esRef.current = es
    es.onmessage = e => {
      if (!pausedRef.current) {
        setLines(prev => [...prev.slice(-2000), e.data])
      }
    }
    return () => es.close()
  }, [tab])

  useEffect(() => {
    if (!paused) bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [lines, paused])

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div className="page-header" style={{ marginBottom: 12 }}>
        <h1>Logs</h1>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <div className="tab-group">
            {(['container', 'tracker'] as Tab[]).map(t => (
              <button key={t} className={`tab ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
                {t === 'container' ? 'Container' : 'Tracker CSV'}
              </button>
            ))}
          </div>
          <button className="btn-ghost" onClick={() => setPaused(p => !p)}>
            {paused ? '▶ Resume' : '⏸ Pause'}
          </button>
          <button className="btn-ghost" onClick={() => setLines([])}>Clear</button>
        </div>
      </div>

      <div className="log-box">
        {lines.map((l, i) => <div key={i} className="log-line">{l}</div>)}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
