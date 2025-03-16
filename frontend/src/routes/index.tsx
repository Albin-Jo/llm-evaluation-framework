import React from 'react';
import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import { DashboardLayout } from '@/layouts/dashboard-layout';
import { DashboardPage } from '@/features/dashboard/dashboard-page';
import { DatasetList } from '@/features/datasets/dataset-list';
import { DatasetDetail } from '@/features/datasets/dataset-detail';
import { DatasetCreate } from '@/features/datasets/dataset-create';
import { DatasetEdit } from '@/features/datasets/dataset-edit';
import { DatasetPreview } from '@/features/datasets/dataset-preview';
import { PlaceholderPage } from '@/features/placeholders/placeholder-page';

const router = createBrowserRouter([
  {
    path: '/',
    element: <DashboardLayout />,
    children: [
      // Dashboard
      {
        index: true,
        element: <DashboardPage />,
      },

      // Datasets
      {
        path: 'datasets',
        element: <DatasetList />,
      },
      {
        path: 'datasets/new',
        element: <DatasetCreate />,
      },
      {
        path: 'datasets/:id',
        element: <DatasetDetail />,
      },
      {
        path: 'datasets/:id/edit',
        element: <DatasetEdit />,
      },
      {
        path: 'datasets/:id/preview',
        element: <DatasetPreview />,
      },

      // Placeholder pages for other modules
      {
        path: 'evaluations/*',
        element: <PlaceholderPage />,
      },
      {
        path: 'microagents/*',
        element: <PlaceholderPage />,
      },
      {
        path: 'prompts/*',
        element: <PlaceholderPage />,
      },
      {
        path: 'comparisons/*',
        element: <PlaceholderPage />,
      },
      {
        path: 'settings/*',
        element: <PlaceholderPage />,
      },

      // 404 page - catch all
      {
        path: '*',
        element: <div className="p-6 text-center">
          <h1 className="text-4xl font-bold mb-4">404</h1>
          <p className="text-lg text-muted-foreground mb-6">Page not found</p>
        </div>,
      },
    ],
  },
]);

export function AppRoutes() {
  return <RouterProvider router={router} />;
}