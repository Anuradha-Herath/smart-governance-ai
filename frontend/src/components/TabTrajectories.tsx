import React, { useState, useEffect } from 'react'
import { Search, TrendingUp, AlertTriangle, History } from 'lucide-react'
import type { Policy, HistoricalDecision, TrajectoryData, IndicatorTrajectory } from '../types'

interface TabTrajectoriesProps {
  selectedPolicy: Policy | null
}

export const TabTrajectories: React.FC<TabTrajectoriesProps> = ({ selectedPolicy }) => {
  const [searchQuery, setSearchQuery] = useState('Proposal for revision of electricity tariff 2023 CEB')
  const [searchResults, setSearchResults] = useState<HistoricalDecision[]>([])
  const [searching, setSearching] = useState(false)

  const [trajectoryData, setTrajectoryData] = useState<TrajectoryData | null>(null)
  const [loadingTraj, setLoadingTraj] = useState(false)
  const [selectedIndicator, setSelectedIndicator] = useState<string>('')

  const year = selectedPolicy?.year ?? 2023

  useEffect(() => {
    fetchTrajectories()
  }, [year])

  useEffect(() => {
    if (searchQuery.trim().length >= 3) {
      handleSearch()
    }
  }, [])

  const fetchTrajectories = async () => {
    setLoadingTraj(true)
    try {
      const res = await fetch(`/api/trajectories?year=${year}&window_years=2`)
      if (res.ok) {
        const data: TrajectoryData = await res.json()
        setTrajectoryData(data)
        const keys = Object.keys(data.trajectories || {})
        if (keys.length > 0 && (!selectedIndicator || !data.trajectories[selectedIndicator])) {
          setSelectedIndicator(keys[0])
        }
      }
    } catch (err) {
      console.error('Failed to load trajectories:', err)
    } finally {
      setLoadingTraj(false)
    }
  }

  const handleSearch = async () => {
    if (!searchQuery.trim()) return
    setSearching(true)
    try {
      const res = await fetch(`/api/historical/search?q=${encodeURIComponent(searchQuery.trim())}&top_k=3`)
      if (res.ok) {
        const data = await res.json()
        setSearchResults(data.results || [])
      }
    } catch (err) {
      console.error('Failed to search historical decisions:', err)
    } finally {
      setSearching(false)
    }
  }

  const currentTraj: IndicatorTrajectory | undefined =
    trajectoryData?.trajectories?.[selectedIndicator]

  // Render SVG Chart for indicator
  const renderChart = (traj: IndicatorTrajectory) => {
    const points = traj.time_series || []
    if (points.length === 0) return null

    const values = points.map((p) => p.value)
    const minVal = Math.min(...values)
    const maxVal = Math.max(...values)
    const range = maxVal - minVal || 1

    const width = 700
    const height = 240
    const paddingX = 60
    const paddingY = 40

    const chartW = width - paddingX * 2
    const chartH = height - paddingY * 2

    const coords = points.map((p, idx) => {
      const x = paddingX + (idx / (points.length - 1)) * chartW
      const y = height - paddingY - ((p.value - minVal) / range) * chartH
      return { ...p, x, y }
    })

    const pathD = coords.reduce((acc, c, idx) => {
      return idx === 0 ? `M ${c.x} ${c.y}` : `${acc} L ${c.x} ${c.y}`
    }, '')

    // Find x of policy event year
    const eventPoint = coords.find((c) => c.year === year)

    return (
      <div className="w-full overflow-x-auto">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full max-w-2xl h-auto mx-auto select-none overflow-visible"
        >
          {/* Grid lines */}
          {[0, 0.5, 1].map((pct, idx) => {
            const y = height - paddingY - pct * chartH
            const labelVal = (minVal + pct * range).toFixed(1)
            return (
              <g key={idx}>
                <line
                  x1={paddingX}
                  y1={y}
                  x2={width - paddingX}
                  y2={y}
                  stroke="#334155"
                  strokeDasharray="4 4"
                  strokeWidth="1"
                />
                <text
                  x={paddingX - 10}
                  y={y + 4}
                  fill="#94a3b8"
                  fontSize="11"
                  textAnchor="end"
                >
                  {labelVal}
                </text>
              </g>
            )
          })}

          {/* Policy Decision Vertical Line */}
          {eventPoint && (
            <g>
              <line
                x1={eventPoint.x}
                y1={paddingY - 10}
                x2={eventPoint.x}
                y2={height - paddingY}
                stroke="#ef4444"
                strokeWidth="2"
                strokeDasharray="5 5"
              />
              <text
                x={eventPoint.x}
                y={paddingY - 16}
                fill="#fca5a5"
                fontSize="11"
                fontWeight="bold"
                textAnchor="middle"
              >
                Decision Year ({year})
              </text>
            </g>
          )}

          {/* Line series */}
          <path d={pathD} fill="none" stroke="#38bdf8" strokeWidth="3" />

          {/* Nodes */}
          {coords.map((c, idx) => (
            <g key={idx}>
              <circle
                cx={c.x}
                cy={c.y}
                r={c.year === year ? '6' : '4'}
                fill={c.year === year ? '#ef4444' : '#0284c7'}
                stroke="#0f172a"
                strokeWidth="2"
              />
              <text
                x={c.x}
                y={c.y - 12}
                fill="#f8fafc"
                fontSize="11"
                fontWeight="bold"
                textAnchor="middle"
              >
                {c.value.toFixed(1)}
              </text>
              <text
                x={c.x}
                y={height - paddingY + 20}
                fill="#cbd5e1"
                fontSize="12"
                textAnchor="middle"
              >
                {c.year}
              </text>
            </g>
          ))}
        </svg>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Causal Disclaimer Banner */}
      <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 text-xs md:text-sm text-amber-200 leading-relaxed flex items-start gap-3">
        <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
        <div>
          <strong className="text-amber-300 font-semibold">Causal Inference Boundary:</strong> Observed indicator shifts (e.g., inflation deceleration from 49.7% to 16.5%) are reported as temporal trajectories rather than pure causal effects of tariff revisions, strictly accounting for concurrent macroeconomic shocks (currency float, debt default, IMF agreement).
        </div>
      </div>

      {/* Live Historical Cabinet Decision Retrieval */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm space-y-4">
        <div>
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <History className="w-4 h-4 text-sky-400" />
            Live Historical Cabinet Decision Retrieval Engine
          </h3>
          <p className="text-xs text-slate-400">
            Search across 11,175 Sri Lanka Cabinet Decisions (2010–2026) using semantic TF-IDF scoring.
          </p>
        </div>

        <div className="flex gap-2">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
            placeholder="Enter policy query in English or Sinhala..."
            className="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-3.5 py-2 text-xs md:text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-sky-500"
          />
          <button
            onClick={handleSearch}
            disabled={searching}
            className="px-4 py-2 bg-sky-600 hover:bg-sky-500 text-white rounded-lg text-xs md:text-sm font-semibold transition flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
          >
            <Search className="w-4 h-4" />
            {searching ? 'Searching...' : 'Search'}
          </button>
        </div>

        {/* Results */}
        <div className="space-y-3 pt-2">
          {searchResults.map((r) => (
            <div
              key={r.doc_id}
              className="bg-slate-950/70 border border-slate-800/80 rounded-lg p-4 space-y-2 text-xs md:text-sm"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="px-2.5 py-0.5 rounded-full bg-sky-500/15 border border-sky-500/30 text-sky-400 font-bold text-[11px]">
                  Match: {r.relevance_tier} (Score: {r.similarity_score.toFixed(3)})
                </span>
                <span className="text-slate-400 text-xs">
                  Date: <strong>{r.date_str}</strong> &nbsp;|&nbsp; ID: <code>{r.doc_id}</code>
                </span>
              </div>
              <h4 className="text-sm font-bold text-slate-100">{r.title}</h4>
              <p className="text-slate-300 text-xs leading-relaxed">{r.body_snippet}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Trajectories Section */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm space-y-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-sky-400" />
              Macroeconomic & Sector Trajectories ({year - 2} to {year + 2})
            </h3>
            <p className="text-xs text-slate-400">
              Symmetric 5-year observation window centered at decision year {year}.
            </p>
          </div>

          {/* Indicator Dropdown */}
          {trajectoryData?.trajectories && (
            <select
              value={selectedIndicator}
              onChange={(e) => setSelectedIndicator(e.target.value)}
              className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs md:text-sm text-slate-100 focus:outline-none focus:border-sky-500 cursor-pointer"
            >
              {Object.entries(trajectoryData.trajectories).map(([code, item]) => (
                <option key={code} value={code}>
                  {code} — {item.indicator_name}
                </option>
              ))}
            </select>
          )}
        </div>

        {/* Chart */}
        {loadingTraj ? (
          <div className="p-8 text-center text-slate-400 text-sm">Loading economic trajectories...</div>
        ) : currentTraj ? (
          <div>
            {renderChart(currentTraj)}

            {/* Trajectory Metrics */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-5 border-t border-slate-800 mt-4">
              <div className="bg-slate-950/70 border border-slate-800 rounded-lg p-3">
                <span className="block text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-1">
                  Baseline ({currentTraj.baseline_year})
                </span>
                <span className="text-lg font-bold text-white">
                  {currentTraj.baseline_value ?? 'N/A'}
                </span>
              </div>

              <div className="bg-slate-950/70 border border-slate-800 rounded-lg p-3">
                <span className="block text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-1">
                  Decision Year ({year})
                </span>
                <span className="text-lg font-bold text-sky-400">
                  {currentTraj.event_value ?? 'N/A'}
                </span>
              </div>

              <div className="bg-slate-950/70 border border-slate-800 rounded-lg p-3">
                <span className="block text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-1">
                  Trend & Change
                </span>
                <span className="text-lg font-bold text-emerald-400 flex items-center gap-1">
                  {currentTraj.trend_direction} ({currentTraj.pct_change > 0 ? '+' : ''}{currentTraj.pct_change?.toFixed(2)}%)
                </span>
              </div>
            </div>
          </div>
        ) : (
          <div className="p-8 text-center text-slate-500 text-sm">No trajectory data available.</div>
        )}
      </div>

      {/* Verified Confounding Macro Shocks */}
      {trajectoryData?.confounding_events && trajectoryData.confounding_events.length > 0 && (
        <div className="space-y-3 pt-2">
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-400" />
            Verified Confounding Macroeconomic Shocks ({year - 2} to {year + 2})
          </h3>

          <div className="space-y-2">
            {trajectoryData.confounding_events.map((c, idx) => (
              <div
                key={idx}
                className="bg-slate-900 border border-slate-800 border-l-4 border-l-amber-500 rounded-lg p-3.5 text-xs md:text-sm space-y-1.5"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="font-bold text-slate-100">
                    {c.year} ({c.date_period}) — {c.event_title}
                  </span>
                  <span className="px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 font-semibold text-[11px]">
                    {c.category}
                  </span>
                </div>
                <p className="text-slate-300 leading-relaxed">{c.description}</p>
                <div className="text-slate-400 text-xs italic">
                  Impact Direction: {c.impact_direction}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
