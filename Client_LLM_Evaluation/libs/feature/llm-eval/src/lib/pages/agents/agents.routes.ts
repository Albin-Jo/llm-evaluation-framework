import { Routes } from '@angular/router';
import { AuthGuard } from '@ngtx-apps/utils/services';
import { AgentsPage } from './agents.page';
import { AgentDetailPage } from './agents-detail/agents-detail.page';
import { AgentCreateEditPage } from './agents-create-edit/agents-create-edit.page';
import { AgentTestPage } from './agents-test/agents-test.page';

export const agentsRoutes: Routes = [
  {
    path: '',
    component: AgentsPage,
    title: 'Agents',
    canActivate: [AuthGuard]
  },
  {
    path: 'create',
    component: AgentCreateEditPage,
    title: 'Create Agent',
    canActivate: [AuthGuard]
  },
  {
    path: ':id',
    component: AgentDetailPage,
    title: 'Agent Details',
    canActivate: [AuthGuard]
  },
  {
    path: ':id/edit',
    component: AgentCreateEditPage,
    title: 'Edit Agent',
    canActivate: [AuthGuard]
  },
  {
    path: ':id/test',
    component: AgentTestPage,
    title: 'Test Agent',
    canActivate: [AuthGuard]
  }
];
