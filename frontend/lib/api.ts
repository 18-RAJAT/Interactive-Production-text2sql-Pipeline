import axios, { AxiosError } from "axios";
import type { GenerateSQLRequest, GenerateSQLResponse, APIError } from "@/types/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const client = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: { "Content-Type": "application/json" },
});

export async function generateSQL(
  payload: GenerateSQLRequest
): Promise<GenerateSQLResponse> {
  try {
    const { data } = await client.post<GenerateSQLResponse>(
      "/generate_sql",
      payload
    );
    return data;
  } catch (err) {
    const error = err as AxiosError<{ detail?: string }>;
    const message =
      error.response?.data?.detail ||
      error.message ||
      "Failed to generate SQL";
    const apiError: APIError = {
      message,
      status: error.response?.status,
    };
    throw apiError;
  }
}

export async function checkHealth(): Promise<boolean> {
  try {
    await client.get("/health", { timeout: 3000 });
    return true;
  } catch {
    return false;
  }
}