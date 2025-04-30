export interface ErrorWithContext {
  name: string;
  appId: string;
  time: any;
  url: string;
  status: number;
  message: string;
  stack?: string; // Optional property
  code?: string;  // Optional property
  response?: any; // Adjust type as needed
}
