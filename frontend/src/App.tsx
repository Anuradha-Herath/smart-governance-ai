import React, { useState, useEffect } from 'react'
import {
  LayoutDashboard,
  MessageSquare,
  BookOpen,
  TrendingUp,
  Sliders,
  AlertCircle,
} from 'lucide-react'
import { Navbar } from './components/Navbar'
import { PolicyHero } from './components/PolicyHero'
import { TabExecutive } from './components/TabExecutive'
import { TabPublicVoice } from './components/TabPublicVoice'
import { TabExpertEvidence } from './components/TabExpertEvidence'
import { TabTrajectories } from './components/TabTrajectories'
import { TabSimulator } from './components/TabSimulator'
import type { Policy, SynthesisData } from './types'

export const App: React.FC = () => {
  const [policies, setPolicies] = useState<Policy[]>([])
  const [selectedPolicyId, setSelectedPolicyId] = useState<string>('DEC_2023_ELEC_01')
  const [synthesis, setSynthesis] = useState<SynthesisData | null>(null)
  const [activeTab, setActiveTab] = useState<'exec' | 'p1' | 'p2' | 'p3' | 'sim'>('exec')
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadInitialData()
  }, [])

  const loadInitialData = async () => {
    try {
      setLoading(true)
      const [polRes, synthRes] = await Promise.all([
        fetch('/api/policies'),
        fetch('/api/synthesis'),
      ])

      if (polRes.ok) {
        const polData = await polRes.json()
        setPolicies(polData.policies || [])
        if (polData.default) setSelectedPolicyId(polData.default)
      }

      if (synthRes.ok) {
        const synthData = await synthRes.json()
        setSynthesis(synthData)
      }
    } catch (err: any) {
      console.error('Failed to initialize app data:', err)
      setError('Failed to connect to the backend API. Please make sure the FastAPI server is running on port 8000.')
    } finally {
      setLoading(false)
    }
  }

  const handleExportMarkdown = async () => {
    try {
      const res = await fetch('/api/briefing/markdown')
      if (res.ok) {
        const data = await res.json()
        const blob = new Blob([data.markdown], { type: 'text/markdown' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `policy_briefing_${selectedPolicyId}.md`
        a.click()
        URL.revokeObjectURL(url)
      }
    } catch (err) {
      console.error('Failed to export markdown briefing:', err)
    }
  }

  const selectedPolicy = policies.find((p) => p.id === selectedPolicyId) || policies[0] || null
  const isPrimary = selectedPolicyId === 'DEC_2023_ELEC_01'

  const tabs = [
    { id: 'exec', label: 'Executive Synthesis', icon: LayoutDashboard },
    { id: 'p1', label: `Public Voice (${selectedPolicyId})`, icon: MessageSquare },
    { id: 'p2', label: 'Expert Evidence', icon: BookOpen },
    {
      id: 'p3',
      label: `Macro Trajectories (${(selectedPolicy?.year || 2023) - 2}–${(selectedPolicy?.year || 2023) + 2})`,
      icon: TrendingUp,
    },
    { id: 'sim', label: 'Live Stance Simulator', icon: Sliders },
  ]

  return (
    <div className="min-h-screen bg-[#0B0F19] text-slate-100 flex flex-col font-sans selection:bg-sky-500/30 selection:text-sky-200">
      <Navbar onExportMarkdown={handleExportMarkdown} />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 md:px-6 py-6">
        {error && (
          <div className="mb-6 p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-300 text-sm flex items-center gap-3">
            <AlertCircle className="w-5 h-5 shrink-0 text-red-400" />
            <div>
              <strong>Connection Notice:</strong> {error}
            </div>
          </div>
        )}

        {/* Hero Showcase Banner */}
        <PolicyHero
          policies={policies}
          selectedPolicy={selectedPolicy}
          onSelectPolicy={(id) => setSelectedPolicyId(id)}
        />

        {/* Segmented Pill Navigation Tab Bar */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-1.5 shadow-md mb-6 flex flex-wrap gap-1.5">
          {tabs.map((t) => {
            const Icon = t.icon
            const isActive = activeTab === t.id
            return (
              <button
                key={t.id}
                onClick={() => setActiveTab(t.id as any)}
                className={`flex-1 min-w-[150px] flex items-center justify-center gap-2 px-3 py-2.5 rounded-lg text-xs md:text-sm font-semibold transition-all duration-200 cursor-pointer ${
                  isActive
                    ? 'bg-gradient-to-r from-sky-500/25 to-sky-600/20 text-white border border-sky-400/80 shadow-[0_0_12px_rgba(56,189,248,0.2)]'
                    : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/60 border border-transparent'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-sky-400' : 'text-slate-500'}`} />
                <span className="truncate">{t.label}</span>
              </button>
            )
          })}
        </div>

        {/* Tab Contents */}
        {loading ? (
          <div className="py-20 text-center text-slate-400 text-sm">
            Initializing Smart Governance AI Dashboard...
          </div>
        ) : (
          <div>
            {activeTab === 'exec' && (
              <TabExecutive synthesis={synthesis} isPrimary={isPrimary} />
            )}
            {activeTab === 'p1' && (
              <TabPublicVoice synthesis={synthesis} selectedPolicy={selectedPolicy} />
            )}
            {activeTab === 'p2' && (
              <TabExpertEvidence synthesis={synthesis} />
            )}
            {activeTab === 'p3' && (
              <TabTrajectories selectedPolicy={selectedPolicy} />
            )}
            {activeTab === 'sim' && (
              <TabSimulator selectedPolicy={selectedPolicy} />
            )}
          </div>
        )}
      </main>

      <footer className="border-t border-slate-900 bg-slate-950/80 py-4 text-center text-xs text-slate-500">
        Smart Governance AI Decision Support Platform • Grounded Multilingual Synthesis Framework
      </footer>
    </div>
  )
}

export default App
