/* Path: libs/feature/llm-eval/src/lib/pages/reports/reports.routes.ts */
import { Routes } from '@angular/router';
import { ReportsPage } from './reports.page';
import { AuthGuard } from '@ngtx-apps/utils/services';

export const REPORTS_ROUTES: Routes = [
  {
    path: '',
    canActivate: [AuthGuard],
    loadComponent: () =>
      import('./reports.page').then((m) => m.ReportsPage)
  }
];
