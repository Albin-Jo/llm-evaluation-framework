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
  pass_threshold: number;
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
  pass_threshold: number;
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
  results?: Record<string, any>[]; // Changed to array for ngFor compatibility
  metrics_results?: Record<string, number>;
  progress?: EvaluationProgress;
  error_message?: string;
  start_time?: string; // Added for template compatibility
  end_time?: string; // Added for template compatibility
  experiment_id?: string; // Added for template compatibility
  pass_threshold?: string;
}

/**
 * Interface for evaluation progress - Updated to match backend response
 */
export interface EvaluationProgress {
  // Core progress fields that match backend
  total: number;
  completed: number;
  failed: number;
  percentage: number;

  // Template compatibility fields (mapped from backend)
  percentage_complete: number; // Maps to progress_percentage
  processed_items: number; // Maps to completed_items
  total_items: number; // Maps to total_items

  // Time-related fields
  estimated_completion?: string; // Calculated from estimated_time_remaining_seconds
  eta_seconds?: number; // Maps to estimated_time_remaining_seconds

  // Status and metadata
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
