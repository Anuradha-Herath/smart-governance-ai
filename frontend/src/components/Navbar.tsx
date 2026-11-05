import React from 'react'
import { Landmark, Shield, Download } from 'lucide-react'

interface NavbarProps {
  onExportMarkdown: () => void
}

export const Navbar: React.FC<NavbarProps> = ({ onExportMarkdown }) => {
  return (
    <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50 px-6 py-3">
      <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-sky-500/10 border border-sky-500/30 text-sky-400">
            <Landmark className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-bold text-slate-100 tracking-tight">Smart Governance AI</h1>
              <span className="text-xs px-2 py-0.5 rounded-full bg-sky-500/15 border border-sky-500/30 text-sky-400 font-semibold uppercase tracking-wider flex items-center gap-1">
                <Shield className="w-3 h-3" /> National Platform
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Evidence-Based Decision Support: Multilingual Sentiment, Verifiable Citations & Historical Outcomes
            </p>
          </div>
        </div>

        {/* Pipeline Status Badges & Actions */}
        <div className="flex items-center gap-3">
          <div className="hidden md:flex items-center gap-2 bg-slate-950/60 px-3 py-1.5 rounded-lg border border-slate-800 text-xs text-slate-300">
            <span className="flex items-center gap-1.5 font-medium text-emerald-400">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> P1 Public (Online)
            </span>
            <span className="text-slate-600">|</span>
            <span className="flex items-center gap-1.5 font-medium text-emerald-400">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> P2 Grounded (99.88%)
            </span>
            <span className="text-slate-600">|</span>
            <span className="flex items-center gap-1.5 font-medium text-emerald-400">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> P3 Precedents (Online)
            </span>
          </div>

          <button
            onClick={onExportMarkdown}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-sky-600 hover:bg-sky-500 text-white text-xs font-semibold shadow-sm transition"
          >
            <Download className="w-3.5 h-3.5" />
            Export Briefing (.md)
          </button>
        </div>
      </div>
    </header>
  )
}
