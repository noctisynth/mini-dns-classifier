import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';

import { createBrowserRouter, RouterProvider } from 'react-router';
import Main from './views/main';
import { Report } from './views/report';
import List from './views/list';

const router = createBrowserRouter([
  {
    path: '/',
    Component: Main,
  },
  {
    path: '/list',
    Component: List,
  },
  {
    path: '/report/:reportId',
    Component: Report,
    loader: async ({ params }) => {
      const { reportId } = params;
      if (!reportId) {
        throw new Error('Report ID is required');
      }
      return { reportId };
    },
  },
]);

const container = document.getElementById('root') as HTMLElement;
const root = createRoot(container);
root.render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);
