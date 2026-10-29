export interface Policy {
  id: string
  label: string
  title: string
  year: number
  date: string
  badge: string
  domain: string
  desc: string
  meta: Array<{ key: string; value: string }>
  vline: string
}

export interface Tension {
  tension_name: string
  public_perspective: string
  expert_perspective: string
  governance_resolution: string
}

export interface Recommendation {
  action_title: string
  detail: string
  evidence_backing: string
}

export interface GrievanceTheme {
  theme: string
  description: string
  prevalence: string
  sample_quote: string
  language: string
  comment_id: string
}

export interface Citation {
  quote: string
  speaker_title: string
  doc_id: string
  statement_id?: string
}

export interface SynthesisData {
  case_id: string
  case_title: string
  enactment_date: string
  executive_summary: string
  pillar1_public_opinion: {
    total_policy_comments_analyzed: number
    news_broadcasts_count: number
    stance_distribution_pct: Record<string, number>
    sentiment_distribution_pct: Record<string, number>
    sampling_disclaimer: string
    grievance_themes?: GrievanceTheme[]
    by_decision?: Record<string, any>
  }
  pillar2_expert_evidence: {
    policy_relevant_statements: number
    verbatim_grounding_rate: string
    top_citations: {
      empirical_evidence?: Citation[]
      systemic_risks?: Citation[]
      expected_benefits?: Citation[]
      recommendations?: Citation[]
    }
  }
  pillar3_historical_indicators: {
    causal_disclaimer: string
  }
  cross_pipeline_tensions: Tension[]
  evidence_based_recommendations: Recommendation[]
}

export interface CommentItem {
  comment_id: string
  text: string
  language: string
  sentiment: string
  stance?: string
  channel?: string
  video_publish_date?: string
  video_title?: string
  video_url?: string
  decision_id?: string
}

export interface StatementItem {
  statement_id: string
  statement_type: string
  source_type?: string
  date?: string
  doc_id: string
  speaker_or_title: string
  extracted_statement: string
  confidence: number
  grounding_verified: boolean
}

export interface HistoricalDecision {
  doc_id: string
  date_str: string
  title: string
  body_snippet: string
  similarity_score: number
  relevance_tier: string
}

export interface TrajectorySeriesItem {
  year: number
  value: number
}

export interface IndicatorTrajectory {
  indicator_name: string
  indicator_code: string
  baseline_year: number
  baseline_value: number
  event_value: number
  trend_direction: string
  pct_change: number
  time_series: TrajectorySeriesItem[]
}

export interface ConfoundingEvent {
  year: number
  date_period: string
  event_title: string
  category: string
  description: string
  impact_direction: string
}

export interface TrajectoryData {
  decision_year: number
  window_years: number
  trajectories: Record<string, IndicatorTrajectory>
  confounding_events: ConfoundingEvent[]
}

export interface StancePrediction {
  stance: string
  sentiment: string
  stance_confidence: number
  sentiment_confidence: number
  sampling_disclaimer: string
}
