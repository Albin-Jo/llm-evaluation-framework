import { Routes } from '@angular/router';
import { DatasetsPage } from './datasets.page';
import { DatasetDetailPage } from './dataset-detail/dataset-detail.page';
import { DatasetUploadPage } from './dataset-upload/dataset-upload.page';

export const datasetsRoutes: Routes = [
  {
    path: '',
    component: DatasetsPage,
  },
  {
    path: 'upload',
    component: DatasetUploadPage,
  },
  {
    path: ':id',
    component: DatasetDetailPage,
  },
];
