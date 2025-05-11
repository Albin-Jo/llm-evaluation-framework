
/**
 * Report status types
 */
export enum ReportStatus {
  DRAFT = 'draft',
  GENERATED = 'generated',
  SENT = 'sent',
  FAILED = 'failed'
}

/**
 * Report format types
 */
export enum ReportFormat {
  PDF = 'pdf',
  HTML = 'html',
  JSON = 'json'
}

/**
 * Base Report properties
 */
export interface Report {
  id: string;
  name: string;
  description?: string;
  status: ReportStatus;
  format: ReportFormat;
  content?: Record<string, any>;
  config?: Record<string, any>;
  file_path?: string;
  last_sent_at?: string;
  evaluation_id: string;
  created_at: string;
  updated_at: string;
  // is_public?: boolean;
}

/**
 * Data for creating a new Report
 */
export interface ReportCreate {
  name: string;
  description?: string;
  evaluation_id: string;
  format: ReportFormat;
  config?: Record<string, any>;
  include_executive_summary?: boolean;
  include_evaluation_details?: boolean;
  include_metrics_overview?: boolean;
  include_detailed_results?: boolean;
  include_agent_responses?: boolean;
  max_examples?: number;
}

/**
 * Data for updating an existing Report
 */
export interface ReportUpdate {
  name?: string;
  description?: string;
  format?: ReportFormat;
  config?: Record<string, any>;
  // is_public?: boolean;
  status?: ReportStatus;
}

/**
 * Response from Report operations
 */
export interface ReportResponse extends Report {}

/**
 * Parameters for filtering Reports
 */
export interface ReportFilterParams {
  skip?: number;
  limit?: number;
  page?: number;
  evaluation_id?: string;
  status?: ReportStatus;
  // is_public?: boolean;
  name?: string;
  format?: ReportFormat;
  sortBy?: 'name' | 'status' | 'format' | 'created_at' | 'updated_at';
  sortDirection?: 'asc' | 'desc';
}

/**
 * Response for listing Reports
 */
export interface ReportListResponse {
  reports: Report[];
  totalCount: number;
}

/**
 * Response for detailed Report with evaluation summary
 */
export interface ReportDetailResponse extends Report {
  evaluation_summary?: Record<string, any>;
}

/**
 * Email recipient interface
 */
export interface EmailRecipient {
  email: string;
  name?: string;
}

/**
 * Request for sending a report via email
 */
export interface SendReportRequest {
  recipients: EmailRecipient[];
  subject?: string;
  message?: string;
  include_pdf?: boolean;
}
