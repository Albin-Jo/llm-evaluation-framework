import { Routes } from '@angular/router';
import { PromptsPage } from './prompts.page';
import { PromptDetailPage } from './prompt-detail/prompt-detail.page';
import { PromptCreateEditPage } from './prompt-create-edit/prompt-create-edit.page';

export const PROMPTS_ROUTES: Routes = [
  {
    path: '',
    component: PromptsPage,
    title: 'Prompts'
  },
  {
    path: 'create',
    component: PromptCreateEditPage,
    title: 'Create Prompt'
  },
  {
    path: ':id',
    component: PromptDetailPage,
    title: 'Prompt Details'
  },
  {
    path: ':id/edit',
    component: PromptCreateEditPage,
    title: 'Edit Prompt'
  }
];
