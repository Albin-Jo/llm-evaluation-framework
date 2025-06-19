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
  MANUAL = 'manual',
}

/**
 * Evaluation status types
 */
export enum EvaluationStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
}

/**
 * Interface for impersonation validation request
 */
export interface ImpersonationValidationRequest {
  employee_id: string;
}

/**
 * Interface for impersonation validation response
 */
export interface ImpersonationValidationResponse {
  valid: boolean;
  employee_id: string;
  user_info?: {
    employee_id: string;
    name: string;
    preferred_username: string;
    email: string;
    expires_in: number;
    token_type: string;
    scope: string;
  };
  user_display?: string;
  token_expires_in?: number;
  validation_timestamp?: string;
  message?: string;
  error?: string;
}

/**
 * Interface for impersonation context in evaluation results
 */
export interface ImpersonationContext {
  is_impersonated: boolean;
  impersonated_user_id?: string;
  impersonated_user_info?: {
    employee_id: string;
    name: string;
    preferred_username: string;
    email: string;
    expires_in: number;
    token_type: string;
    scope: string;
    session_state?: string;
  };
  impersonated_user_display?: string;
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
  pass_threshold?: number;
  impersonate_user_id?: string; // New field for user impersonation
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
  pass_threshold?: number;
  impersonate_user_id?: string; // New field for user impersonation
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
  sortBy?: string;
  sortDirection?: 'asc' | 'desc';
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
  passed?: boolean; // Added for pass/fail status
  [key: string]: any;
}

/**
 * Interface for evaluation results summary
 */
export interface EvaluationResultsSummary {
  pass_rate: number;
  pass_count: number;
  total_evaluated: number;
  pass_threshold: number;
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
  pass_threshold?: number; // New field
  impersonate_user_id?: string; // New field
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
  results?: Record<string, any>[];
  metrics_results?: Record<string, number>;
  progress?: EvaluationProgress;
  error_message?: string;
  impersonation_context?: ImpersonationContext; // New field for impersonation details
}

/**
 * Interface for evaluation progress
 */
export interface EvaluationProgress {
  total: number;
  completed: number;
  failed: number;
  percentage: number;
  percentage_complete: number;
  processed_items: number;
  total_items: number;
  estimated_completion?: string;
  eta_seconds?: number;
  status?: EvaluationStatus;
  start_time?: string;
  last_updated?: string;
  running_time_seconds?: number;
}

/**
 * Interface for backend progress response (for reference)
 */
export interface BackendProgressResponse {
  status: string;
  total_items: number;
  completed_items: number;
  progress_percentage: number;
  start_time: string;
  end_time?: string | null;
  running_time_seconds: number;
  estimated_time_remaining_seconds: number;
  last_updated: string;
}

/**
 * Interface for evaluation list response
 */
export interface EvaluationsResponse {
  evaluations: Evaluation[];
  totalCount: number;
}

/**
 * Interface for evaluation results response with enhanced metadata
 */
export interface EvaluationResultsResponse {
  items: EvaluationResult[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
  has_previous: boolean;
  summary: EvaluationResultsSummary;
  impersonation_context: ImpersonationContext;
  evaluation_info: {
    id: string;
    name: string;
    method: string;
    status: string;
  };
}
