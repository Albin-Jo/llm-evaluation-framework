
/**
 * Defines the structure for creating a new prompt
 */
export interface PromptCreate {
  name: string;
  description?: string;
  content: string;
  parameters?: Record<string, any>;
  version?: string;
  is_public?: boolean;
  template_id?: string;
  category?: string;
}

/**
 * Defines the structure for prompt response from API
 */
export interface PromptResponse {
  id: string;
  name: string;
  description: string | null;
  content: string;
  parameters: Record<string, any>;
  version: string;
  is_public: boolean;
  template_id: string | null;
  created_at: string;
  updated_at: string;
  category?: string;
}

/**
 * Defines the structure for updating an existing prompt
 */
export interface PromptUpdate {
  name?: string;
  description?: string;
  content?: string;
  parameters?: Record<string, any>;
  version?: string;
  is_public?: boolean;
  template_id?: string;
  category?: string;
}

/**
 * Defines the filter options for prompt listing
 */
export interface PromptFilter {
  skip?: number;
  limit?: number;
  is_public?: boolean;
  template_id?: string;
}


export interface Prompt {
  id: string;
  name: string;
  description?: string;
  template: string;
  variables?: string[];
  created_at: string;
  updated_at: string;
  created_by?: string;
  tags?: string[];
  version?: number;
}


/**
 * Interface for prompt filtering parameters
 */
 export interface PromptFilterParams {
  name?: string;
  category?: string;
  isPublic?: boolean;
  page?: number;
  limit?: number;
  sortBy?: string;
  sortDirection?: 'asc' | 'desc';
}

/**
 * Interface for prompts list response
 */
 export interface PromptsResponse {
  prompts: PromptResponse[];
  totalCount: number;
}
