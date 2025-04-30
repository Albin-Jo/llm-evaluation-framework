
//Common PATHS
const COMMON_PATHS = {
  EMPTY: '',
}

/**
 * Define all the Route Paths here
 */
export const PATHS = {
  //Global Routes
  ...COMMON_PATHS,
  APP: 'app',
  FORBIDDEN: 'forbidden',
  UNATHORIZED: 'unauthorized',
  CALLBACK: 'callback',
  LOGOUT: 'logout',
  NOT_FOUND: '**'
};


  //Project specific Routes
export const FEATURE_PATHS = {
  ...COMMON_PATHS,
  PROJECTS: 'projects',
  TASKS: 'tasks',
  HOME: 'home',
  DASHBOARD: 'dashboard',
  PRICING: 'pricing',
  PROFILE: 'profile',
  SUB1: 'sub1',
  DATASETS: 'datasets',
  PROMPTS: 'prompts',
  LLM_EVAL: 'llm-eval',
  AGENTS: 'agents',
  EVALUATIONS: 'evaluations',
  REPORTS: 'reports'
}


