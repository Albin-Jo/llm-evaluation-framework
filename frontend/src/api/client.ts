import axios from 'axios';

// Define base URL from environment variable with a fallback
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 seconds
});

// Add request interceptor for auth token, etc.
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    // Handle common errors
    const { response } = error;

    if (!response) {
      // Network error
      console.error('Network error, please check your connection');
    } else if (response.status === 401) {
      // Auth error - redirect to login
      localStorage.removeItem('token');
      window.location.href = '/login';
    } else if (response.status === 404) {
      // Resource not found
      console.error('Resource not found');
    } else if (response.status >= 500) {
      // Server error
      console.error('Server error, please try again later');
    }

    return Promise.reject(error);
  }
);

export default apiClient;