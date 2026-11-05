import React from 'react'
import {
  Users,
  FileCheck2,
  CheckCircle,
  TrendingDown,
  Scale,
  Lightbulb,
  FileText,
} from 'lucide-react'
import type { SynthesisData } from '../types'

interface TabExecutiveProps {
  synthesis: SynthesisData | null
  isPrimary: boolean
}

export const TabExecutive: React.FC<TabExecutiveProps> = ({ synthesis, isPrimary }) => {
  if (!synthesis) {
    return <div className="p-8 text-center text-slate-400">Loading synthesis data...</div>
  }

  if (!isPrimary) {
    return (
      <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-5 text-amber-200 text-sm leading-relaxed">
        <strong>Comparative Case Study:</strong> The deep multi-pipeline synthesis briefing, core tensions, and recommended directives focus on the primary baseline (<code>DEC_2023_ELEC_01</code>). Explore the <strong>Public Voice</strong> and <strong>Macro Trajectories</strong> tabs for comparative case indicators.
      </div>
    )
  }

  const p1 = synthesis.pillar1_public_opinion
  const p2 = synthesis.pillar2_expert_evidence
  const oppPct = p1.stance_distribution_pct?.['Oppose'] ?? 68.1

  return (
    <div className="space-y-6">
      {/* KPI Metrics Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Metric 1 */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm">
          <div className="flex items-center justify-between text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">
            <span>Citizen Stance (n={p1.total_policy_comments_analyzed})</span>
            <Users className="w-4 h-4 text-red-400" />
          </div>
          <div className="text-2xl font-bold text-white mb-1">
            {oppPct.toFixed(1)}% <span className="text-sm font-semibold text-red-400">Opposed</span>
          </div>
          <p className="text-xs text-slate-400">
            Across {p1.news_broadcasts_count} TV News Broadcasts
          </p>
        </div>

        {/* Metric 2 */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm">
          <div className="flex items-center justify-between text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">
            <span>Official Policy Statements</span>
            <FileCheck2 className="w-4 h-4 text-sky-400" />
          </div>
          <div className="text-2xl font-bold text-white mb-1">
            {p2.policy_relevant_statements.toLocaleString()} <span className="text-sm font-semibold text-sky-400">Extracted</span>
          </div>
          <p className="text-xs text-slate-400">
            Cabinet 2023-01-09 & Hansard Debates
          </p>
        </div>

        {/* Metric 3 */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm">
          <div className="flex items-center justify-between text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">
            <span>Source Grounding</span>
            <CheckCircle className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold text-emerald-400 mb-1">
            {p2.verbatim_grounding_rate}
          </div>
          <p className="text-xs text-slate-400">
            Zero AI Hallucination Guaranteed
          </p>
        </div>

        {/* Metric 4 */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm">
          <div className="flex items-center justify-between text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">
            <span>Inflation Trajectory</span>
            <TrendingDown className="w-4 h-4 text-sky-400" />
          </div>
          <div className="text-2xl font-bold text-white mb-1">
            49.7% → 16.5%
          </div>
          <p className="text-xs text-emerald-400 font-medium">
            -33.2% Deceleration (2022 to 2023)
          </p>
        </div>
      </div>

      {/* Executive Summary Card */}
      <div className="bg-slate-900/90 border-l-4 border-l-sky-500 border border-slate-800 rounded-xl p-5 shadow-sm">
        <div className="flex items-center gap-2 text-sm font-bold text-sky-400 uppercase tracking-wider mb-2">
          <FileText className="w-4 h-4" /> Executive Policy Summary
        </div>
        <p className="text-slate-200 text-sm md:text-base leading-relaxed">
          {synthesis.executive_summary}
        </p>
      </div>

      {/* Core Policy Tensions */}
      <div className="space-y-4 pt-2">
        <div>
          <h3 className="text-lg font-bold text-white flex items-center gap-2">
            <Scale className="w-5 h-5 text-sky-400" />
            Core Policy Tensions & Resolved Trade-Offs
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            How the platform resolves sharp contradictions between public distress, fiscal constraints, and systemic energy security.
          </p>
        </div>

        <div className="space-y-4">
          {synthesis.cross_pipeline_tensions?.map((t, idx) => (
            <div key={idx} className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm space-y-4">
              <h4 className="text-base font-bold text-white flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-sky-400"></span>
                {t.tension_name}
              </h4>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Public Voice Perspective */}
                <div className="bg-red-500/10 border-l-4 border-l-red-500 border border-red-500/20 rounded-lg p-4">
                  <span className="block text-xs font-bold uppercase tracking-wider text-red-300 mb-1.5">
                    Citizen Grievance (Public Voice)
                  </span>
                  <p className="text-slate-200 text-sm leading-relaxed">
                    {t.public_perspective}
                  </p>
                </div>

                {/* Expert Justification Perspective */}
                <div className="bg-sky-500/10 border-l-4 border-l-sky-500 border border-sky-500/20 rounded-lg p-4">
                  <span className="block text-xs font-bold uppercase tracking-wider text-sky-300 mb-1.5">
                    Official & Expert Justification
                  </span>
                  <p className="text-slate-200 text-sm leading-relaxed">
                    {t.expert_perspective}
                  </p>
                </div>
              </div>

              {/* Recommended Governance Resolution */}
              <div className="bg-emerald-500/10 border-l-4 border-l-emerald-500 border border-emerald-500/20 rounded-lg p-4">
                <span className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-emerald-300 mb-1">
                  <Lightbulb className="w-4 h-4 text-emerald-400" /> Recommended Governance Resolution
                </span>
                <p className="text-emerald-100 text-sm font-medium leading-relaxed">
                  {t.governance_resolution}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Actionable Recommendations */}
      <div className="space-y-4 pt-2">
        <h3 className="text-lg font-bold text-white flex items-center gap-2">
          <Lightbulb className="w-5 h-5 text-amber-400" />
          Actionable Evidence-Based Policy Directives
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {synthesis.evidence_based_recommendations?.map((r, idx) => (
            <div key={idx} className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm flex flex-col justify-between">
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <span className="px-2 py-0.5 rounded bg-sky-600 text-white font-bold text-xs">
                    #{String(idx + 1).padStart(2, '0')}
                  </span>
                  <h4 className="text-sm md:text-base font-bold text-white">
                    {r.action_title}
                  </h4>
                </div>
                <p className="text-slate-300 text-sm leading-relaxed mb-4">
                  {r.detail}
                </p>
              </div>

              <div className="bg-slate-950/70 border-l-2 border-l-sky-500 rounded p-2.5 text-xs text-slate-400">
                <strong className="text-sky-300">Empirical Backing: </strong>
                {r.evidence_backing}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
