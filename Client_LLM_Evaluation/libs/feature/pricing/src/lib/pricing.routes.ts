import { Routes } from '@angular/router';
import { FEATURE_PATHS } from '@ngtx-apps/utils/shared';
import { QompeatPage } from './pages/qompeat/qompeat.page';
import { Sub1Page } from './pages/sub1/sub1.page';
import { Sub2Page } from './pages/sub2/sub2.page';
import { Sub3Page } from './pages/sub3/sub3.page';
import { Sub4Page } from './pages/sub4/sub4.page';


export const pricingRoutes: Routes = [
  {
    path: '',
    children: [
      {
        path: FEATURE_PATHS.SUB1,
        title: 'Sub 1',
        component: Sub1Page
      },
      {
        path: 'sub2',
        title: 'Sub 2',
        component: Sub2Page
      },
      {
        path: 'sub3',
        title: 'Sub 3',
        component: Sub3Page
      },
      {
        path: 'sub4',
        title: 'Sub 4',
        component: Sub4Page
      },
      {
        path: FEATURE_PATHS.EMPTY,
        redirectTo: '/app/pricing/sub1',
        pathMatch: 'full',
      },
    ]
  },

];
