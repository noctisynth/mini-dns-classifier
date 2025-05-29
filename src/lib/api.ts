import axios from 'axios';

export const axiosInstance = axios.create({
  baseURL: import.meta.env.API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface PredictResult {
  timestamp: string;
  query: string;
  prediction: 'Covert Channel' | 'Normal Traffic';
  confidence: number;
  confidence_percent: string;
}

export interface AnalysisReport {
  total_packets: number;
  dns_packets: number;
  valid_queries: number;
  covert_count: number;
  normal_count: number;
  top_covert_queries: PredictResult[];
}

export interface PredictResponse {
  message: string;
  report_id: string;
  data: {
    report: AnalysisReport;
    created_at: string;
  };
}

export async function predict(file: File): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append('file', file);
  const response = await axiosInstance.put<PredictResponse>(
    '/api/predict',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    },
  );
  if (response.status !== 200) {
    throw new Error(`Error: ${response.status} - ${response.statusText}`);
  }
  return response.data;
}

export interface Report {
  message: string;
  data: {
    id: string;
    report: AnalysisReport;
    csv_path?: string;
    created_at: string;
  };
}

export async function getReport(reportId: string): Promise<Report> {
  const response = await axiosInstance.get(`/api/report/${reportId}`);
  if (response.status !== 200) {
    throw new Error(`Error: ${response.status} - ${response.statusText}`);
  }
  return response.data;
}

export interface Reports {
  message: string;
  data: {
    id: string;
    report: AnalysisReport;
    created_at: string;
  }[];
}

export async function getReports(): Promise<Reports> {
  const response = await axiosInstance.get('/api/reports/');
  if (response.status !== 200) {
    throw new Error(`Error: ${response.status} - ${response.statusText}`);
  }
  return response.data;
}

export async function getReportCSV(reportId: string): Promise<Blob> {
  const response = await axiosInstance.get(`/api/report/${reportId}/download`, {
    responseType: 'blob',
  });
  if (response.status !== 200) {
    throw new Error(`Error: ${response.status} - ${response.statusText}`);
  }
  return response.data;
}
