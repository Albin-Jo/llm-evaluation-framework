import { Routes } from '@angular/router';
import { FEATURE_PATHS } from '@ngtx-apps/utils/shared';
import { ProjectsPage } from './pages/projects/projects.page';
import { TasksPage } from './pages/tasks/tasks.page';

export const dashboardRoutes: Routes = [
  {
    path: FEATURE_PATHS.EMPTY,
    children: [
      {
        path: FEATURE_PATHS.PROJECTS,
        component: ProjectsPage
      },
      {
        path: FEATURE_PATHS.TASKS,
        component: TasksPage
      },
      {
        path: FEATURE_PATHS.EMPTY,
        redirectTo: '/app/dashboard/projects',
        pathMatch: 'full',
      },
    ]
  },

];
