import { Routes } from '@angular/router';
import { AuthGuard } from '@ngtx-apps/utils/services';

export const COMPARISONS_ROUTES: Routes = [
  {
    path: '',
    canActivate: [AuthGuard],
    loadComponent: () =>
      import('./comparisons.page').then((m) => m.ComparisonsPage),
  },
  {
    path: 'create',
    canActivate: [AuthGuard],
    loadComponent: () =>
      import('./comparison-create-edit/comparison-create-edit.page').then(
        (m) => m.ComparisonCreateEditPage
      ),
    data: { title: 'Create Comparison' },
  },
  {
    path: ':id',
    canActivate: [AuthGuard],
    loadComponent: () =>
      import('./comparison-detail/comparison-detail.page').then(
        (m) => m.ComparisonDetailPage
      ),
    data: { title: 'Comparison Details' },
  },
  {
    path: ':id/edit',
    canActivate: [AuthGuard],
    loadComponent: () =>
      import('./comparison-create-edit/comparison-create-edit.page').then(
        (m) => m.ComparisonCreateEditPage
      ),
    data: { title: 'Edit Comparison' },
  },
];