/* Path: libs/feature/llm-eval/src/lib/pages/evaluations/evaluations.routes.ts */
import { Routes } from '@angular/router';
import { AuthGuard } from '@ngtx-apps/utils/services';

export const EVALUATIONS_ROUTES: Routes = [
  {
    path: '',
    canActivate: [AuthGuard],
    loadComponent: () =>
      import('./evaluations.page').then((m) => m.EvaluationsPage),
  },
  {
    path: 'create',
    canActivate: [AuthGuard],
    loadComponent: () =>
      import('./evaluation-create-edit/evaluation-create-edit.page').then(
        (m) => m.EvaluationCreateEditPage
      ),
    data: { title: 'Create Evaluation' }
  },
  {
    path: ':id',
    canActivate: [AuthGuard],
    loadComponent: () =>
      import('./evaluation-detail/evaluation-detail.page').then(
        (m) => m.EvaluationDetailPage
      ),
    data: { title: 'Evaluation Details' }
  },
  {
    path: ':id/edit',
    canActivate: [AuthGuard],
    loadComponent: () =>
      import('./evaluation-create-edit/evaluation-create-edit.page').then(
        (m) => m.EvaluationCreateEditPage
      ),
    data: { title: 'Edit Evaluation' }
  }
];