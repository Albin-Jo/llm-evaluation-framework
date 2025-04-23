import { Routes } from '@angular/router';
import { LayoutPage } from './layout.page';
import { FEATURE_PATHS } from '@ngtx-apps/utils/shared';

export const layoutRoutes: Routes = [
  {
    path: FEATURE_PATHS.EMPTY,
    component: LayoutPage,
    children: [
      {
        path: FEATURE_PATHS.HOME,
        title: 'Home',
        loadComponent: () => import('@ngtx-apps/feature/home').then(m => m.HomePage),
      },
      {
        path: FEATURE_PATHS.DASHBOARD,
        title: 'Dashboard',
        loadChildren: () => import('@ngtx-apps/feature/dashboard').then(m => m.dashboardRoutes),
      },
      {
        path: FEATURE_PATHS.PRICING,
        title: 'Pricing',
        loadChildren: () => import('@ngtx-apps/feature/pricing').then(m => m.pricingRoutes),
      },
      {
        path: FEATURE_PATHS.PROFILE,
        title: 'Profile',
        loadComponent: () => import('@ngtx-apps/feature/profile').then(m => m.ProfilePage),
      },
      {
        path: FEATURE_PATHS.SUB1,
        title: 'Sub 1',
        loadComponent: () => import('@ngtx-apps/feature/sub1').then(m => m.Sub1Page),
      },
    {
        path: FEATURE_PATHS.DATASETS,
        loadChildren: () => import('@ngtx-apps/feature/llm-eval').then(m => m.llmEvalRoutes)
      },
      {
        path: FEATURE_PATHS.EMPTY,
        redirectTo: '/app/home',
        pathMatch: 'full',

      },
    {
        path: FEATURE_PATHS.PROMPTS,
        loadChildren: () => import('@ngtx-apps/feature/llm-eval').then(m => m.PROMPTS_ROUTES)
      },
    {
        path: FEATURE_PATHS.AGENTS,
        loadChildren: () => import('@ngtx-apps/feature/llm-eval').then(m => m.agentsRoutes)
      },
      {
        path: FEATURE_PATHS.EVALUATIONS,
        loadChildren: () => import('@ngtx-apps/feature/llm-eval').then(m => m.EVALUATIONS_ROUTES)
      },
    ]
  },

];
