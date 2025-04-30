/* Path: libs/feature/llm-eval/src/lib/llm-eval.routes.ts */
import { Routes } from '@angular/router';
import { AuthGuard } from '@ngtx-apps/utils/services';

export const llmEvalRoutes: Routes = [
  // Redirect the root path to datasets
  {
    path: '',
    redirectTo: 'datasets',
    pathMatch: 'full'
  },
  // Load dataset routes directly, not as children
  {
    path: 'datasets',
    loadChildren: () => {
      console.log('Loading dataset routes');
      return import('./pages/datasets/datasets.routes').then(m => {
        console.log('Dataset routes loaded:', m);
        return m.datasetsRoutes;
      });
    }
  },
  {
    path: 'comparisons',
    loadChildren: () => import('./pages/comparisons/comparisons.routes').then(m => m.comparisonsRoutes),
    canActivate: [AuthGuard]
  }
];
