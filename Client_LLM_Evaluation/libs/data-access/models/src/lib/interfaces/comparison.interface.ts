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
 * Interface for metric difference
 */
export interface MetricDifference {
  id: string;
  name: string;
  evaluation_a_value: number;
  evaluation_b_value: number;
  absolute_difference: number;
  percentage_difference: number;
  is_improvement: boolean;
  is_significant: boolean;
  meta_info?: Record<string, any>;
  comparison_id: string;
  created_at: string;
  updated_at: string;
}

/**
 * Interface for sample difference
 */
export interface SampleDifference {
  id: string;
  sample_id: string;
  evaluation_a_score?: number;
  evaluation_b_score?: number;
  absolute_difference?: number;
  percentage_difference?: number;
  status: 'improved' | 'regressed' | 'unchanged';
  comparison_id: string;
  created_at: string;
  updated_at: string;
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
  summary?: Record<string, any>;
  created_at: string;
  updated_at: string;
  created_by_id?: string;
}

/**
 * Interface for detailed comparison response
 */
export interface ComparisonDetail extends Comparison {
  evaluation_a?: Record<string, any>;
  evaluation_b?: Record<string, any>;
  metric_differences?: MetricDifference[];
  result_differences?: Record<string, SampleDifference[]>;
  summary?: {
    overall_result?: number;
    total_metrics?: number;
    improved_metrics?: number;
    regressed_metrics?: number;
    unchanged_metrics?: number;
    significant_changes?: number;
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
