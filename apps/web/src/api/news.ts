import type { DashboardRequest, DashboardResponse } from "../types/dashboard";

const API_URL = "http://127.0.0.1:8000/api/news/dashboard";

export async function fetchDashboard(
  request: DashboardRequest,
): Promise<DashboardResponse> {
  let response: Response;

  try {
    response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });
  } catch {
    throw new Error("Backend is not running at 127.0.0.1:8000.");
  }

  if (!response.ok) {
    throw new Error("NewsBoardAI could not create a dashboard for that topic.");
  }

  return response.json();
}
