/**
 * Comparison status types
 */
export enum ComparisonStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
}

/**
 * Interface for comparison creation
 */
export interface ComparisonCreate {
  name: string;
  description?: string;
  evaluation_a_id: string;
  evaluation_b_id: string;
  config?: Record<string, any>;
}

/**
 * Interface for comparison updating
 */
export interface ComparisonUpdate {
  name?: string;
  description?: string;
  config?: Record<string, any>;
}

/**
 * Interface for comparison filtering parameters
 */
export interface ComparisonFilterParams {
  name?: string;
  page?: number;
  limit?: number;
  status?: ComparisonStatus;
  evaluation_a_id?: string;
  evaluation_b_id?: string;
  sortBy?: string;
  sortDirection?: 'asc' | 'desc';
}

/**
 * Interface for metric difference - aligned with API response
 */
export interface MetricDifference {
  metric_name?: string; // Added to match API
  name?: string; // For backwards compatibility
  evaluation_a_value: number;
  evaluation_b_value: number;
  absolute_difference: number;
  percentage_change?: number; // Renamed from percentage_difference
  percentage_difference?: number; // Keep for backwards compatibility
  is_improvement: boolean;
  is_significant?: boolean;
}

/**
 * Interface for sample difference - aligned with API response
 */
export interface SampleDifference {
  sample_id: string;
  evaluation_a_score?: number;
  evaluation_b_score?: number;
  absolute_difference?: number;
  percentage_difference?: number;
  status: 'improved' | 'regressed' | 'unchanged';
  input_data?: Record<string, any>;
  evaluation_a_output?: Record<string, any>;
  evaluation_b_output?: Record<string, any>;
}

/**
 * Interface for comparison response
 */
export interface Comparison {
  id: string;
  name: string;
  description?: string;
  evaluation_a_id: string;
  evaluation_b_id: string;
  status: ComparisonStatus;
  config?: Record<string, any>;
  comparison_results?: Record<string, any>;
  summary?: ComparisonSummary;
  created_at: string;
  updated_at: string;
  created_by_id?: string;
  narrative_insights?: string; // Added based on API response
}

/**
 * Interface for comparison summary - aligned with API response
 */
export interface ComparisonSummary {
  evaluation_a_name?: string;
  evaluation_b_name?: string;
  evaluation_a_method?: string;
  evaluation_b_method?: string;
  overall_result?: string; // "improved", "regressed", etc.
  percentage_change?: number; // The actual percentage improvement/regression
  total_metrics?: number;
  improved_metrics?: number;
  regressed_metrics?: number;
  unchanged_metrics?: number;
  improved_samples?: number;
  regressed_samples?: number;
  matched_samples?: number;
  significant_improvements?: number;
  significant_regressions?: number;
  significance_rate?: number;
  metric_improvement_rate?: number;
  weighted_improvement_score?: number; // Added based on API response
  consistency_score?: number; // Added based on API response
  cross_method_comparison?: boolean; // Added based on API response
  top_improvements?: Array<{
    metric_name: string;
    absolute_difference: number;
    percentage_change: number;
    is_significant?: boolean;
    effect_size?: number;
    effect_magnitude?: string;
  }>;
  top_regressions?: Array<{
    metric_name: string;
    absolute_difference: number;
    percentage_change: number;
    is_significant?: boolean;
    effect_size?: number;
    effect_magnitude?: string;
  }>;
  statistical_power?: {
    sample_size: number;
    power_category: string;
    recommendations: string[];
    significant_results: number;
    significance_rate: number;
  };
  effect_size_summary?: {
    metric_effect_sizes: any[];
    average_effect_size: number | null;
  };
}

/**
 * Interface for detailed comparison response
 */
export interface ComparisonDetail extends Comparison {
  evaluation_a?: Record<string, any>;
  evaluation_b?: Record<string, any>;
  metric_differences?: MetricDifference[];
  result_differences?: Record<string, SampleDifference[]>;
  metric_configs?: Record<
    string,
    {
      higher_is_better: boolean;
      weight: number;
      description?: string;
    }
  >;
  overall_comparison?: {
    overall_scores?: {
      evaluation_a: number;
      evaluation_b: number;
      absolute_difference: number;
      percentage_change: number;
      is_improvement: boolean;
    };
    metric_stats?: {
      total_metrics: number;
      improved_metrics: number;
      regressed_metrics: number;
      metric_improvement_rate: number;
    };
  };
}

/**
 * Interface for comparison list response
 */
export interface ComparisonsResponse {
  comparisons: Comparison[];
  totalCount: number;
}

/**
 * Interface for visualization data
 */
export interface VisualizationData {
  type: 'radar' | 'bar' | 'line';
  labels: string[];
  datasets: {
    label: string;
    data: number[];
    backgroundColor?: string;
    borderColor?: string;
    fill?: boolean;
  }[];
  options?: Record<string, any>;
}
