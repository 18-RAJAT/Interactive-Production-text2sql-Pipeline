export interface GenerateSQLRequest {
  question: string;
  schema: string;
}

export interface GenerateSQLResponse {
  generated_sql: string;
  confidence?: number;
  execution_result?: Record<string, unknown>[];
  latency_ms?: number;
}

export interface QueryHistoryItem {
  id: string;
  question: string;
  schema: string;
  generated_sql: string;
  confidence?: number;
  timestamp: number;
}

export interface APIError {
  message: string;
  status?: number;
}