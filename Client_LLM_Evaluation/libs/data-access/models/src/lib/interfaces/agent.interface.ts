/* Path: libs/data-access/models/src/lib/interfaces/agent.interface.ts */

/**
 * Available agent status options
 */
export enum AgentStatus {
  ACTIVE = 'active',
  INACTIVE = 'inactive'
}

/**
 * Agent domain types
 */
export enum AgentDomain {
  GENERAL = 'general',
  CUSTOMER_SERVICE = 'customer_service',
  TECHNICAL = 'technical',
  MEDICAL = 'medical',
  LEGAL = 'legal',
  FINANCE = 'finance',
  EDUCATION = 'education',
  OTHER = 'other'
}

/**
 * Base Agent properties
 */
export interface Agent {
  id: string;
  name: string;
  description?: string;
  api_endpoint: string;
  domain: string;
  config?: Record<string, any>;
  is_active: boolean;
  model_type?: string;
  version?: string;
  tags?: string[];
  created_at: string;
  updated_at: string;
  created_by_id?: string;
}

/**
 * Data for creating a new Agent
 */
export interface AgentCreate {
  name: string;
  description?: string;
  api_endpoint: string;
  domain: string;
  config?: Record<string, any>;
  is_active?: boolean;
  model_type?: string;
  version?: string;
  tags?: string[];
}

/**
 * Data for updating an existing Agent
 */
export interface AgentUpdate {
  name?: string;
  description?: string;
  api_endpoint?: string;
  domain?: string;
  config?: Record<string, any>;
  is_active?: boolean;
  model_type?: string;
  version?: string;
  tags?: string[];
}

/**
 * Response from Agent operations
 */
export interface AgentResponse extends Agent {}

/**
 * Parameters for filtering Agents
 */
export interface AgentFilterParams {
  skip?: number;
  limit?: number;
  domain?: string;
  is_active?: boolean;
  name?: string;
  sortBy?: 'name' | 'domain' | 'created_at' | 'updated_at';
  sortDirection?: 'asc' | 'desc';
  page?: number;
}

/**
 * Response for listing Agents
 */
export interface AgentListResponse {
  agents: Agent[];
  totalCount: number;
}
