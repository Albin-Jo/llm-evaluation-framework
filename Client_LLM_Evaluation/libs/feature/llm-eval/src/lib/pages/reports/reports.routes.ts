/* Path: libs/feature/llm-eval/src/lib/pages/reports/reports.routes.ts */
import { Routes } from '@angular/router';
import { AuthGuard } from '@ngtx-apps/utils/services';

export const REPORTS_ROUTES: Routes = [
  {
    path: '',
    canActivate: [AuthGuard],
    loadComponent: () =>
      import('./reports.page').then((m) => m.ReportsPage)
  },
  {
    path: 'create',
    canActivate: [AuthGuard],
    loadComponent: () =>
      import('./report-create-edit/report-create-edit.page').then(
        (m) => m.ReportCreateEditPage
      ),
    data: { title: 'Create Report' }
  },
  {
    path: ':id',
    canActivate: [AuthGuard],
    loadComponent: () =>
      import('./report-detail/report-detail.page').then(
        (m) => m.ReportDetailPage
      ),
    data: { title: 'Report Details' }
  },
  {
    path: ':id/edit',
    canActivate: [AuthGuard],
    loadComponent: () =>
      import('./report-create-edit/report-create-edit.page').then(
        (m) => m.ReportCreateEditPage
      ),
    data: { title: 'Edit Report' }
  },
  {
    path: ':id/preview',
    canActivate: [AuthGuard],
    loadComponent: () =>
      import('./report-preview/report-preview.page').then(
        (m) => m.ReportPreviewPage
      ),
    data: { title: 'Report Preview' }
  }
];
