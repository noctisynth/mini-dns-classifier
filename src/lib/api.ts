import axios from 'axios';

export const axiosInstance = axios.create({
  baseURL: import.meta.env.API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface PredictResponse {
  message: string;
  report_id: string;
  data: unknown;
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
