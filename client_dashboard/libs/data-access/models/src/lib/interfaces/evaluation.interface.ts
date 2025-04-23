/* Path: libs/data-access/models/src/lib/interfaces/evaluation.interface.ts */
import { Agent } from './agent.interface';
import { Dataset } from './dataset.interface';
import { Prompt } from './prompt.interface';

/**
 * Evaluation method types
 */
export enum EvaluationMethod {
  RAGAS = 'ragas',
  DEEPEVAL = 'deepeval',
  CUSTOM = 'custom',
  MANUAL = 'manual'
}

/**
 * Evaluation status types
 */
export enum EvaluationStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled'
}

/**
 * Interface for evaluation creation
 */
export interface EvaluationCreate {
  name: string;
  description?: string;
  method: EvaluationMethod;
  config?: Record<string, any>;
  metrics?: string[];
  agent_id: string;
  dataset_id: string;
  prompt_id: string;
}

/**
 * Interface for evaluation updating
 */
export interface EvaluationUpdate {
  name?: string;
  description?: string;
  status?: EvaluationStatus;
  config?: Record<string, any>;
  metrics?: string[];
  experiment_id?: string;
  start_time?: string;
  end_time?: string;
}

/**
 * Interface for evaluation filtering parameters
 */
export interface EvaluationFilterParams {
  name?: string;
  page?: number;
  limit?: number;
  status?: EvaluationStatus;
  agent_id?: string;
  dataset_id?: string;
  prompt_id?: string;
  method?: EvaluationMethod;
  sortBy?: string; // Added for template compatibility
  sortDirection?: 'asc' | 'desc'; // Added for template compatibility
}

/**
 * Interface for evaluation metric score
 */
export interface MetricScore {
  id: string;
  name: string;
  value: number;
  weight: number;
  meta_info?: Record<string, any>;
  result_id: string;
  created_at: string;
  updated_at: string;
}

/**
 * Interface for evaluation result
 */
export interface EvaluationResult {
  id: string;
  overall_score?: number;
  raw_results?: Record<string, any>;
  dataset_sample_id?: string;
  input_data?: Record<string, any>;
  output_data?: Record<string, any>;
  processing_time_ms?: number;
  evaluation_id: string;
  created_at: string;
  updated_at: string;
  metric_scores: MetricScore[];
  [key: string]: any;
}

/**
 * Interface for evaluation response
 */
export interface Evaluation {
  id: string;
  name: string;
  description?: string;
  method: EvaluationMethod;
  status: EvaluationStatus;
  config?: Record<string, any>;
  metrics?: string[];
  experiment_id?: string;
  start_time?: string;
  end_time?: string;
  agent_id: string;
  dataset_id: string;
  prompt_id: string;
  created_at: string;
  updated_at: string;
}

/**
 * Interface for detailed evaluation response
 */
export interface EvaluationDetail extends Evaluation {
  agent?: Agent;
  dataset?: Dataset;
  prompt?: Prompt;
  results?: Record<string, any>[];  // Changed to array for ngFor compatibility
  metrics_results?: Record<string, number>;
  progress?: EvaluationProgress;
  error_message?: string;
  start_time?: string; // Added for template compatibility
  end_time?: string; // Added for template compatibility
  experiment_id?: string; // Added for template compatibility
}

/**
 * Interface for evaluation progress
 */
export interface EvaluationProgress {
  total: number;
  completed: number;
  failed: number;
  percentage: number;
  percentage_complete: number; // Added for template compatibility
  processed_items: number; // Added for template compatibility
  total_items: number; // Added for template compatibility
  estimated_completion?: string; // Added for template compatibility
  eta_seconds?: number;
  status?: EvaluationStatus;
}

/**
 * Interface for evaluation list response
 */
export interface EvaluationsResponse {
  evaluations: Evaluation[];
  totalCount: number;
}
