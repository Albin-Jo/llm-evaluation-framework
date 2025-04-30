/* Path: libs/data-access/models/src/lib/interfaces/report.interface.ts */

/**
 * Available report status options
 */
export enum ReportStatus {
  DRAFT = 'draft',
  GENERATED = 'generated'
}

/**
 * Available report format options
 */
export enum ReportFormat {
  PDF = 'pdf',
  HTML = 'html',
  MARKDOWN = 'markdown',
  JSON = 'json'
}

/**
 * Base Report properties
 */
export interface Report {
  id: string;
  name: string;
  description?: string;
  format: ReportFormat;
  config?: Record<string, any>;
  evaluation_id: string;
  status: ReportStatus;
  file_path?: string;
  content?: Record<string, any> | null;
  last_sent_at?: string | null;
  created_at: string;
  updated_at: string;
}

/**
 * Data for creating a new Report
 */
export interface ReportCreate {
  name: string;
  description?: string;
  format: ReportFormat;
  config?: Record<string, any>;
  evaluation_id: string;
}

/**
 * Data for updating an existing Report
 */
export interface ReportUpdate {
  name?: string;
  description?: string;
  format?: ReportFormat;
  config?: Record<string, any>;
}

/**
 * Parameters for filtering Reports
 */
export interface ReportFilterParams {
  skip?: number;
  limit?: number;
  is_public?: boolean;
  name?: string;
  status?: ReportStatus;
  evaluation_id?: string;
  format?: ReportFormat;
  page?: number;
  sortBy?: 'name' | 'created_at' | 'updated_at';
  sortDirection?: 'asc' | 'desc';
}

/**
 * Response for listing Reports
 */
export interface ReportListResponse {
  reports: Report[];
  totalCount: number;
}
