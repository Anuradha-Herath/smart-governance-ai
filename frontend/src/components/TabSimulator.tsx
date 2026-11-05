import React, { useState } from 'react'
import { Play, Sparkles, AlertCircle, Info } from 'lucide-react'
import type { Policy, StancePrediction } from '../types'

interface TabSimulatorProps {
  selectedPolicy: Policy | null
}

export const TabSimulator: React.FC<TabSimulatorProps> = ({ selectedPolicy }) => {
  const [inputText, setInputText] = useState('')
  const [selectedPrompt, setSelectedPrompt] = useState('Custom Input...')
  const [prediction, setPrediction] = useState<StancePrediction | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const samplePrompts = [
    'Custom Input...',
    'විදුලි බිල 66% කින් වැඩි කරපු එක ජනතාවට දරාගන්න බෑ. මිනිස්සුන්ට කන්නවත් සල්ලි නෑ.',
    'CEB bill eka wadi una eka sadaranai, nathnam ape rata thawa thawa anduren andurata yanawa.',
    'Higher tariffs are essential to stop CEB bankrupting the Treasury under the IMF program.',
    'PUCSL should publish generation costs before imposing this burden on small enterprises.',
  ]

  const handleSelectPrompt = (val: string) => {
    setSelectedPrompt(val)
    if (val !== 'Custom Input...') {
      setInputText(val)
    } else {
      setInputText('')
    }
  }

  const handleAnalyze = async () => {
    if (!inputText.trim()) {
      setError('Please enter a comment or statement to analyze.')
      return
    }
    setError(null)
    setLoading(true)

    try {
      const res = await fetch('/api/predict/stance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: inputText.trim(),
          target_decision: selectedPolicy?.id || 'DEC_2023_ELEC_01',
        }),
      })

      if (!res.ok) {
        throw new Error('Inference server returned an error.')
      }

      const data: StancePrediction = await res.json()
      setPrediction(data)
    } catch (err: any) {
      setError(err.message || 'Failed to analyze text.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Intro Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm space-y-4">
        <div>
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-sky-400" />
            Live Multilingual Policy Stance & Sentiment Simulator
          </h3>
          <p className="text-xs text-slate-400 mt-1">
            Test any citizen statement in Sinhala, Singlish, or English against the trained subword TF-IDF classifier.
          </p>
        </div>

        {/* Preset Selector */}
        <div className="space-y-1.5">
          <label className="text-xs font-semibold text-slate-300">
            Choose an example comment or type your own below:
          </label>
          <select
            value={selectedPrompt}
            onChange={(e) => handleSelectPrompt(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3.5 py-2 text-xs md:text-sm text-slate-100 focus:outline-none focus:border-sky-500 cursor-pointer"
          >
            {samplePrompts.map((p, idx) => (
              <option key={idx} value={p}>
                {p}
              </option>
            ))}
          </select>
        </div>

        {/* Text Area */}
        <div className="space-y-1.5">
          <textarea
            value={inputText}
            onChange={(e) => {
              setInputText(e.target.value)
              if (selectedPrompt !== 'Custom Input...') setSelectedPrompt('Custom Input...')
            }}
            placeholder="Type any Sinhala, Singlish, or English comment here..."
            rows={4}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs md:text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-sky-500 leading-relaxed resize-none"
          />
        </div>

        {/* Error message */}
        {error && (
          <div className="text-red-400 text-xs flex items-center gap-1.5 bg-red-500/10 border border-red-500/30 rounded-lg p-2.5">
            <AlertCircle className="w-4 h-4 shrink-0" />
            {error}
          </div>
        )}

        {/* Submit button */}
        <button
          onClick={handleAnalyze}
          disabled={loading}
          className="w-full py-2.5 bg-sky-600 hover:bg-sky-500 text-white font-bold rounded-lg text-xs md:text-sm shadow-md transition flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50"
        >
          <Play className="w-4 h-4 fill-current" />
          {loading ? 'Running Multilingual NLP Pipeline...' : 'Analyze Sentiment & Stance'}
        </button>
      </div>

      {/* Prediction Output Results */}
      {prediction && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Stance Card */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm space-y-3">
              <span className="block text-xs font-bold uppercase tracking-wider text-slate-400">
                Policy Stance Classification
              </span>
              <div>
                <span
                  className={`inline-block px-4 py-1.5 rounded-xl font-bold text-sm tracking-wide ${
                    prediction.stance === 'Oppose'
                      ? 'bg-red-500/20 border border-red-500/40 text-red-400'
                      : prediction.stance === 'Support'
                      ? 'bg-emerald-500/20 border border-emerald-500/40 text-emerald-400'
                      : 'bg-slate-800 text-slate-300'
                  }`}
                >
                  {prediction.stance}
                </span>
              </div>
              <div className="text-xs text-slate-300 pt-2 border-t border-slate-800">
                Classification Confidence: <strong className="text-sky-400">{(prediction.stance_confidence * 100).toFixed(1)}%</strong>
              </div>
            </div>

            {/* Sentiment Card */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm space-y-3">
              <span className="block text-xs font-bold uppercase tracking-wider text-slate-400">
                Sentiment Polarity
              </span>
              <div>
                <span
                  className={`inline-block px-4 py-1.5 rounded-xl font-bold text-sm tracking-wide ${
                    prediction.sentiment === 'Negative'
                      ? 'bg-red-500/20 border border-red-500/40 text-red-400'
                      : prediction.sentiment === 'Positive'
                      ? 'bg-emerald-500/20 border border-emerald-500/40 text-emerald-400'
                      : 'bg-slate-800 text-slate-300'
                  }`}
                >
                  {prediction.sentiment}
                </span>
              </div>
              <div className="text-xs text-slate-300 pt-2 border-t border-slate-800">
                Polarity Confidence: <strong className="text-sky-400">{(prediction.sentiment_confidence * 100).toFixed(1)}%</strong>
              </div>
            </div>
          </div>

          {/* Sampling Disclaimer */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-3.5 flex items-start gap-2.5 text-xs text-slate-400 leading-relaxed">
            <Info className="w-4 h-4 text-sky-400 shrink-0 mt-0.5" />
            <span>{prediction.sampling_disclaimer}</span>
          </div>
        </div>
      )}
    </div>
  )
}
