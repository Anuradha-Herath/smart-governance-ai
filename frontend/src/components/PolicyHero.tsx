import React from 'react'
import { Zap, CheckCircle2, ChevronDown } from 'lucide-react'
import type { Policy } from '../types'

interface PolicyHeroProps {
  policies: Policy[]
  selectedPolicy: Policy | null
  onSelectPolicy: (id: string) => void
}

export const PolicyHero: React.FC<PolicyHeroProps> = ({
  policies,
  selectedPolicy,
  onSelectPolicy,
}) => {
  if (!selectedPolicy) return null

  return (
    <div className="bg-gradient-to-br from-slate-900 via-slate-900 to-slate-950 border border-slate-700/80 rounded-2xl p-6 shadow-xl mb-6 relative overflow-hidden">
      {/* Decorative gradient glow */}
      <div className="absolute top-0 right-0 w-96 h-96 bg-sky-500/5 rounded-full blur-3xl pointer-events-none" />

      {/* Header controls & badges */}
      <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
        <div className="flex flex-wrap items-center gap-2">
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-md bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 text-xs font-bold tracking-wide uppercase">
            <CheckCircle2 className="w-3.5 h-3.5" />
            {selectedPolicy.badge}
          </span>
          <span className="px-2.5 py-1 rounded-md bg-amber-500/15 border border-amber-500/30 text-amber-300 font-mono text-xs font-bold">
            {selectedPolicy.id}
          </span>
          <span className="px-2.5 py-1 rounded-md bg-sky-500/15 border border-sky-500/30 text-sky-300 text-xs font-semibold uppercase tracking-wider">
            {selectedPolicy.domain}
          </span>
        </div>

        {/* Policy Selector Dropdown */}
        <div className="relative">
          <select
            value={selectedPolicy.id}
            onChange={(e) => onSelectPolicy(e.target.value)}
            className="appearance-none bg-slate-800 hover:bg-slate-750 text-slate-100 text-sm font-semibold rounded-lg pl-3.5 pr-9 py-2 border border-slate-700 shadow-sm focus:outline-none focus:border-sky-500 transition cursor-pointer"
          >
            {policies.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
          </select>
          <ChevronDown className="w-4 h-4 text-slate-400 absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none" />
        </div>
      </div>

      {/* Title & Description */}
      <div className="mb-5">
        <h2 className="text-2xl md:text-3xl font-extrabold text-white flex items-center gap-2 tracking-tight">
          <Zap className="w-7 h-7 text-sky-400 shrink-0" />
          {selectedPolicy.title}
        </h2>
        <p className="text-slate-300 text-sm md:text-base mt-2 leading-relaxed max-w-5xl">
          {selectedPolicy.desc}
        </p>
      </div>

      {/* Metadata Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-4 border-t border-slate-800">
        {selectedPolicy.meta.map((m, idx) => (
          <div key={idx} className="bg-slate-950/60 border border-slate-800 rounded-lg p-3">
            <span className="block text-[11px] font-bold text-sky-400 uppercase tracking-wider mb-1">
              {m.key}
            </span>
            <span className="text-slate-200 text-xs md:text-sm font-medium">
              {m.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
