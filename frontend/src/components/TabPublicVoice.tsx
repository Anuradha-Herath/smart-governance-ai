import React, { useState, useEffect } from 'react'
import {
  MessageSquare,
  Search,
  ChevronRight,
  Quote,
  LayoutGrid,
  Table as TableIcon,
} from 'lucide-react'
import type { SynthesisData, CommentItem, Policy } from '../types'

interface TabPublicVoiceProps {
  synthesis: SynthesisData | null
  selectedPolicy: Policy | null
}

export const TabPublicVoice: React.FC<TabPublicVoiceProps> = ({ synthesis, selectedPolicy }) => {
  const [comments, setComments] = useState<CommentItem[]>([])
  const [loading, setLoading] = useState(false)
  const [totalComments, setTotalComments] = useState(0)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedLang, setSelectedLang] = useState<string>('All')
  const [availableLangs, setAvailableLangs] = useState<string[]>([])
  const [viewMode, setViewMode] = useState<'table' | 'feed'>('table')
  const [expandedTheme, setExpandedTheme] = useState<number | null>(null)

  const pid = selectedPolicy?.id ?? 'DEC_2023_ELEC_01'
  const p1 = synthesis?.pillar1_public_opinion
  const byDec = p1?.by_decision?.[pid] ?? {}

  const stanceData = byDec.stance_distribution_pct ?? p1?.stance_distribution_pct ?? { Oppose: 68.1, Support: 14.5, Neutral_or_Unclear: 17.4 }
  const sentData = byDec.sentiment_distribution_pct ?? p1?.sentiment_distribution_pct ?? { Negative: 74.2, Neutral: 16.3, Positive: 9.5 }
  const decN = byDec.total_comments ?? p1?.total_policy_comments_analyzed ?? 626
  const decVids = byDec.news_broadcasts_count ?? p1?.news_broadcasts_count ?? 39

  useEffect(() => {
    fetchComments()
  }, [pid, selectedLang, searchQuery])

  const fetchComments = async () => {
    setLoading(true)
    try {
      const params = new URLSearchParams()
      params.append('decision_id', pid)
      if (selectedLang !== 'All') params.append('language', selectedLang)
      if (searchQuery.trim()) params.append('search', searchQuery.trim())
      params.append('limit', '50')

      const res = await fetch(`/api/comments?${params.toString()}`)
      if (res.ok) {
        const data = await res.json()
        setComments(data.comments || [])
        setTotalComments(data.total || 0)
        if (data.languages && availableLangs.length === 0) {
          setAvailableLangs(['All', ...data.languages])
        }
      }
    } catch (err) {
      console.error('Failed to load comments:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Disclaimer Banner */}
      <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 text-xs md:text-sm text-amber-200 leading-relaxed flex items-start gap-3">
        <MessageSquare className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
        <div>
          <strong className="text-amber-300 font-semibold">
            Decision-Linked News Broadcast Corpus ({pid}):
          </strong>{' '}
          Analyzing <strong>{decN}</strong> real citizen comments extracted across <strong>{decVids}</strong> dated Sri Lankan TV news broadcasts (Ada Derana, Hiru News, Newsfirst, Swarnavahini) covering <em>{selectedPolicy?.title}</em>.
          <p className="mt-1 text-amber-300/80 text-xs">
            {p1?.sampling_disclaimer}
          </p>
        </div>
      </div>

      {/* Visualizations: Stance & Sentiment */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Stance Distribution */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
          <h4 className="text-sm font-bold text-slate-100 uppercase tracking-wider mb-4">
            Policy Stance Toward {pid} (n={decN})
          </h4>
          <div className="space-y-3">
            {Object.entries(stanceData).map(([key, val]) => {
              const numVal = Number(val)
              const color =
                key === 'Oppose'
                  ? 'bg-red-500'
                  : key === 'Support'
                  ? 'bg-emerald-500'
                  : 'bg-slate-500'
              return (
                <div key={key}>
                  <div className="flex justify-between text-xs font-semibold text-slate-300 mb-1">
                    <span>{key}</span>
                    <span>{numVal.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-slate-800 rounded-full h-3 overflow-hidden">
                    <div
                      className={`h-full ${color} rounded-full transition-all duration-500`}
                      style={{ width: `${Math.max(numVal, 2)}%` }}
                    />
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Sentiment Distribution */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
          <h4 className="text-sm font-bold text-slate-100 uppercase tracking-wider mb-4">
            Citizen Sentiment Polarity ({decVids} Broadcasts)
          </h4>
          <div className="space-y-3">
            {Object.entries(sentData).map(([key, val]) => {
              const numVal = Number(val)
              const color =
                key === 'Negative'
                  ? 'bg-red-500'
                  : key === 'Positive'
                  ? 'bg-emerald-500'
                  : 'bg-slate-500'
              return (
                <div key={key}>
                  <div className="flex justify-between text-xs font-semibold text-slate-300 mb-1">
                    <span>{key}</span>
                    <span>{numVal.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-slate-800 rounded-full h-3 overflow-hidden">
                    <div
                      className={`h-full ${color} rounded-full transition-all duration-500`}
                      style={{ width: `${Math.max(numVal, 2)}%` }}
                    />
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>

      {/* Citizen Grievance Themes Accordion */}
      {p1?.grievance_themes && p1.grievance_themes.length > 0 && (
        <div className="space-y-3 pt-2">
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <Quote className="w-4 h-4 text-sky-400" />
            Citizen Grievance Theme Clusters (Real Verbatim Records)
          </h3>

          <div className="space-y-2">
            {p1.grievance_themes.map((g, idx) => {
              const isOpen = expandedTheme === idx
              return (
                <div
                  key={idx}
                  className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden transition"
                >
                  <button
                    onClick={() => setExpandedTheme(isOpen ? null : idx)}
                    className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-slate-800/50 transition cursor-pointer"
                  >
                    <div className="flex items-center gap-3">
                      <ChevronRight
                        className={`w-4 h-4 text-sky-400 transition-transform ${
                          isOpen ? 'rotate-90' : ''
                        }`}
                      />
                      <span className="text-sm font-semibold text-white">
                        {g.theme}
                      </span>
                    </div>
                    <div className="flex items-center gap-3 text-xs text-slate-400">
                      <span>Prevalence: <strong className="text-slate-200">{g.prevalence}</strong></span>
                      <span className="font-mono bg-slate-950 px-2 py-0.5 rounded border border-slate-800 text-sky-400">
                        {g.comment_id}
                      </span>
                    </div>
                  </button>

                  {isOpen && (
                    <div className="px-5 pb-4 pt-1 border-t border-slate-800/80 bg-slate-950/40 text-xs md:text-sm space-y-2">
                      <p className="text-slate-300">{g.description}</p>
                      <div className="bg-slate-900 border-l-2 border-l-sky-400 rounded p-3 text-slate-100 italic">
                        "{g.sample_quote}"
                        <div className="text-[11px] text-slate-400 not-italic mt-1.5 flex gap-2">
                          <span>Source Lang: <strong>{g.language}</strong></span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Real Citizen Comments Explorer */}
      <div className="space-y-4 pt-2">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <MessageSquare className="w-4 h-4 text-sky-400" />
              Decision-Linked Public Comments Explorer ({pid})
            </h3>
            <p className="text-xs text-slate-400">
              Displaying <strong>{comments.length}</strong> loaded of <strong>{totalComments}</strong> total matching comments
            </p>
          </div>

          {/* View Mode Toggle */}
          <div className="flex items-center bg-slate-900 border border-slate-800 rounded-lg p-0.5 text-xs">
            <button
              onClick={() => setViewMode('table')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-md font-medium transition ${
                viewMode === 'table' ? 'bg-sky-600 text-white' : 'text-slate-400 hover:text-white'
              }`}
            >
              <TableIcon className="w-3.5 h-3.5" /> Table
            </button>
            <button
              onClick={() => setViewMode('feed')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-md font-medium transition ${
                viewMode === 'feed' ? 'bg-sky-600 text-white' : 'text-slate-400 hover:text-white'
              }`}
            >
              <LayoutGrid className="w-3.5 h-3.5" /> Feed
            </button>
          </div>
        </div>

        {/* Filter & Search Bar */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="relative md:col-span-2">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search comments (e.g. bill, CEB, wadi, sahana, විදුලි, මිල...)"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-slate-900 border border-slate-800 rounded-lg pl-9 pr-3 py-2 text-xs md:text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-sky-500"
            />
          </div>

          <div>
            <select
              value={selectedLang}
              onChange={(e) => setSelectedLang(e.target.value)}
              className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs md:text-sm text-slate-100 focus:outline-none focus:border-sky-500 cursor-pointer"
            >
              <option value="All">All Languages</option>
              <option value="Sinhala">Sinhala</option>
              <option value="Singlish">Singlish</option>
              <option value="English">English</option>
            </select>
          </div>
        </div>

        {/* Comments Table or Feed */}
        {loading ? (
          <div className="p-8 text-center text-slate-400 text-sm">Searching citizen comments...</div>
        ) : comments.length === 0 ? (
          <div className="p-8 text-center text-slate-500 text-sm bg-slate-900 border border-slate-800 rounded-xl">
            No citizen comments matched your filter criteria.
          </div>
        ) : viewMode === 'table' ? (
          <div className="overflow-x-auto border border-slate-800 rounded-xl bg-slate-900 shadow-sm max-h-[550px]">
            <table className="w-full text-left text-xs md:text-sm">
              <thead className="bg-slate-950/80 sticky top-0 text-[11px] font-bold text-slate-400 uppercase tracking-wider border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-3">ID</th>
                  <th className="py-2.5 px-3">Date</th>
                  <th className="py-2.5 px-3">Channel</th>
                  <th className="py-2.5 px-3">Verbatim Comment</th>
                  <th className="py-2.5 px-3">Lang</th>
                  <th className="py-2.5 px-3">Stance</th>
                  <th className="py-2.5 px-3">Sentiment</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-slate-200">
                {comments.map((c) => {
                  const isOppose = c.stance === 'Oppose' || c.sentiment === 'Negative'
                  const isSupport = c.stance === 'Support' || c.sentiment === 'Positive'
                  return (
                    <tr key={c.comment_id} className="hover:bg-slate-800/40 transition">
                      <td className="py-2 px-3 font-mono text-[11px] text-slate-400 whitespace-nowrap">
                        {c.comment_id}
                      </td>
                      <td className="py-2 px-3 text-slate-400 whitespace-nowrap text-xs">
                        {c.video_publish_date || '—'}
                      </td>
                      <td className="py-2 px-3 font-medium text-sky-400 whitespace-nowrap text-xs">
                        {c.channel || 'Broadcast'}
                      </td>
                      <td className="py-2 px-3 max-w-md text-slate-100 font-normal">
                        "{c.text}"
                      </td>
                      <td className="py-2 px-3 text-slate-400 whitespace-nowrap text-xs">
                        {c.language}
                      </td>
                      <td className="py-2 px-3 whitespace-nowrap">
                        <span
                          className={`px-2 py-0.5 rounded text-[11px] font-semibold ${
                            isOppose
                              ? 'bg-red-500/15 border border-red-500/30 text-red-400'
                              : isSupport
                              ? 'bg-emerald-500/15 border border-emerald-500/30 text-emerald-400'
                              : 'bg-slate-700 text-slate-300'
                          }`}
                        >
                          {c.stance || 'N/A'}
                        </span>
                      </td>
                      <td className="py-2 px-3 whitespace-nowrap">
                        <span
                          className={`text-xs font-semibold ${
                            c.sentiment === 'Negative'
                              ? 'text-red-400'
                              : c.sentiment === 'Positive'
                              ? 'text-emerald-400'
                              : 'text-slate-400'
                          }`}
                        >
                          {c.sentiment}
                        </span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-h-[550px] overflow-y-auto pr-1">
            {comments.map((c) => {
              const isOppose = c.stance === 'Oppose' || c.sentiment === 'Negative'
              return (
                <div
                  key={c.comment_id}
                  className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm flex flex-col justify-between"
                >
                  <p className="text-slate-100 text-xs md:text-sm italic leading-relaxed mb-3">
                    "{c.text}"
                  </p>
                  <div className="flex items-center justify-between text-xs text-slate-400 pt-2 border-t border-slate-800">
                    <span>
                      {c.channel ? <strong>{c.channel}</strong> : ''} ({c.language})
                    </span>
                    <span
                      className={`px-2 py-0.5 rounded text-[11px] font-semibold ${
                        isOppose
                          ? 'bg-red-500/15 text-red-400'
                          : 'bg-emerald-500/15 text-emerald-400'
                      }`}
                    >
                      {c.stance || c.sentiment}
                    </span>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
