
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
 * Integration types for agents
 */
export enum IntegrationType {
  AZURE_OPENAI = 'azure_openai',
  MCP = 'mcp',
  DIRECT_API = 'direct_api',
  CUSTOM = 'custom'
}

/**
 * Authentication types for agents
 */
export enum AuthType {
  API_KEY = 'api_key',
  BEARER_TOKEN = 'bearer_token',
  NONE = 'none'
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
  integration_type?: IntegrationType;
  auth_type?: AuthType;
  auth_credentials?: Record<string, any> | string;
  request_template?: Record<string, any>;
  response_format?: string;
  retry_config?: Record<string, any>;
  content_filter_config?: Record<string, any>;
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
  integration_type?: IntegrationType;
  auth_type?: AuthType;
  auth_credentials?: Record<string, any>;
  request_template?: Record<string, any>;
  response_format?: string;
  retry_config?: Record<string, any>;
  content_filter_config?: Record<string, any>;
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
  integration_type?: IntegrationType;
  auth_type?: AuthType;
  auth_credentials?: Record<string, any>;
  request_template?: Record<string, any>;
  response_format?: string;
  retry_config?: Record<string, any>;
  content_filter_config?: Record<string, any>;
}

/**
 * Response from Agent operations
 */
export interface AgentResponse extends Agent {}

/**
 * Response from Agent health check
 */
export interface AgentHealthResponse {
  status: string;
  healthy: boolean;
  message?: string;
  details?: Record<string, any>;
}

/**
 * Response from Agent tools request
 */
export interface AgentToolsResponse {
  tools: AgentTool[];
}

/**
 * Agent Tool information
 */
export interface AgentTool {
  name: string;
  description: string;
  parameters?: Record<string, any>;
  required_parameters?: string[];
}

/**
 * Parameters for filtering Agents
 */
export interface AgentFilterParams {
  skip?: number;
  limit?: number;
  domain?: string;
  is_active?: boolean;
  name?: string;
  integration_type?: IntegrationType;
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