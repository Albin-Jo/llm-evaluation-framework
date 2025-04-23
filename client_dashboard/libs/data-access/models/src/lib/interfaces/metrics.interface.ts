/* Path: libs/data-access/models/src/lib/interfaces/metrics.interface.ts */
/**
 * Interface for metric details
 */
 export interface MetricDetail {
    name: string;
    description: string;
    min_value?: number;
    max_value?: number;
    default_weight?: number;
  }

  /**
   * Interface for metric category
   */
  export interface MetricCategory {
    name: string;
    description: string;
    metrics: MetricDetail[];
  }

  /**
   * Interface for supported metrics response from API
   */
  export interface SupportedMetricsResponse {
    [category: string]: string[] | string;
  }

  /**
   * Common evaluation metrics for RAGAS
   */
  export const RAGAS_METRICS = {
    'answer_relevance': 'Measures how relevant the answer is to the question',
    'answer_correctness': 'Measures how correct the answer is based on the provided context',
    'answer_completeness': 'Measures how complete the answer is',
    'context_precision': 'Measures how precise the retrieved context is',
    'context_recall': 'Measures how well the context covers the information needed',
    'faithfulness': 'Measures how faithful the answer is to the provided context',
    'context_relevance': 'Measures how relevant the retrieved context is to the question',
    'harmfulness': 'Measures potential harmful content in the answer'
  };

  /**
   * Common evaluation metrics for DeepEval
   */
  export const DEEPEVAL_METRICS = {
    'relevance': 'Measures relevance of response to input',
    'coherence': 'Measures logical flow and structure of the response',
    'fluency': 'Measures linguistic quality and readability of the response',
    'groundedness': 'Measures how well the response is grounded in facts',
    'bias': 'Measures potential bias in the response',
    'toxicity': 'Measures potential toxic content in the response'
  };

  /**
   * Get display name for a metric
   */
  export const getMetricDisplayName = (metric: string): string => {
    // Convert snake_case to Title Case with spaces
    return metric
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  /**
   * Get description for a metric
   */
  export const getMetricDescription = (metric: string): string => {
    return RAGAS_METRICS[metric] ||
           DEEPEVAL_METRICS[metric] ||
           'Measures the performance of the evaluated model';
  };

  /**
   * Format metric value for display
   */
  export const formatMetricValue = (value: number): string => {
    return (value * 100).toFixed(2) + '%';
  };