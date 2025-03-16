import React from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { ChevronLeft, Edit, Trash2, Eye, Download, AlertTriangle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { getDataset } from '@/api/datasets';
import { formatDate } from '@/lib/utils';

export function DatasetDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const { data: dataset, isLoading, error } = useQuery({
    queryKey: ['dataset', id],
    queryFn: () => getDataset(id!),
    enabled: !!id,
  });

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={() => navigate(-1)}>
            <ChevronLeft className="h-4 w-4 mr-1" />
            Back
          </Button>
          <div className="h-8 w-48 bg-muted animate-pulse rounded"></div>
        </div>
        <Card>
          <CardHeader>
            <div className="h-7 w-64 bg-muted animate-pulse rounded mb-2"></div>
            <div className="h-5 w-full bg-muted animate-pulse rounded"></div>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {Array(4).fill(0).map((_, i) => (
                <div key={i} className="space-y-2">
                  <div className="h-5 w-32 bg-muted animate-pulse rounded"></div>
                  <div className="h-8 w-full bg-muted animate-pulse rounded"></div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (error || !dataset) {
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={() => navigate(-1)}>
            <ChevronLeft className="h-4 w-4 mr-1" />
            Back
          </Button>
          <h1 className="text-3xl font-bold tracking-tight">Dataset Details</h1>
        </div>
        <Card className="border-destructive">
          <CardHeader>
            <CardTitle className="flex items-center text-destructive">
              <AlertTriangle className="h-5 w-5 mr-2" />
              Error Loading Dataset
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p>Failed to load dataset details. The dataset may not exist or there was a server error.</p>
          </CardContent>
          <CardFooter>
            <Button onClick={() => navigate('/datasets')}>
              Return to Datasets
            </Button>
          </CardFooter>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={() => navigate(-1)}>
            <ChevronLeft className="h-4 w-4 mr-1" />
            Back
          </Button>
          <h1 className="text-3xl font-bold tracking-tight">{dataset.name}</h1>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" asChild>
            <Link to={`/datasets/${dataset.id}/preview`}>
              <Eye className="h-4 w-4 mr-1" />
              Preview
            </Link>
          </Button>
          <Button variant="outline" size="sm" asChild>
            <Link to={`/datasets/${dataset.id}/edit`}>
              <Edit className="h-4 w-4 mr-1" />
              Edit
            </Link>
          </Button>
          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-1" />
            Download
          </Button>
          <Button variant="destructive" size="sm">
            <Trash2 className="h-4 w-4 mr-1" />
            Delete
          </Button>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Dataset Information</CardTitle>
            {dataset.description && (
              <CardDescription>{dataset.description}</CardDescription>
            )}
          </CardHeader>
          <CardContent>
            <dl className="space-y-4">
              <div>
                <dt className="text-sm font-medium text-muted-foreground">Type</dt>
                <dd className="mt-1">
                  <span className="inline-flex items-center rounded-md bg-blue-50 px-2 py-1 text-xs font-medium text-blue-700 ring-1 ring-inset ring-blue-700/10 dark:bg-blue-400/10 dark:text-blue-400 dark:ring-blue-400/20">
                    {dataset.type}
                  </span>
                </dd>
              </div>
              <div>
                <dt className="text-sm font-medium text-muted-foreground">Version</dt>
                <dd className="mt-1 text-sm">{dataset.version}</dd>
              </div>
              <div>
                <dt className="text-sm font-medium text-muted-foreground">Visibility</dt>
                <dd className="mt-1 text-sm">
                  {dataset.is_public ? 'Public' : 'Private'}
                </dd>
              </div>
              <div>
                <dt className="text-sm font-medium text-muted-foreground">Row Count</dt>
                <dd className="mt-1 text-sm">{dataset.row_count ?? 'Unknown'}</dd>
              </div>
              {dataset.created_at && (
                <div>
                  <dt className="text-sm font-medium text-muted-foreground">Created</dt>
                  <dd className="mt-1 text-sm">{formatDate(dataset.created_at)}</dd>
                </div>
              )}
              {dataset.updated_at && (
                <div>
                  <dt className="text-sm font-medium text-muted-foreground">Last Updated</dt>
                  <dd className="mt-1 text-sm">{formatDate(dataset.updated_at)}</dd>
                </div>
              )}
            </dl>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Schema Information</CardTitle>
            <CardDescription>Structure and metadata about this dataset</CardDescription>
          </CardHeader>
          <CardContent>
            {dataset.schema ? (
              <div className="space-y-4">
                {Object.entries(dataset.schema).map(([key, value]) => (
                  <div key={key}>
                    <dt className="text-sm font-medium text-muted-foreground">{key}</dt>
                    <dd className="mt-1 text-sm overflow-auto max-h-20">
                      {typeof value === 'object'
                        ? JSON.stringify(value, null, 2)
                        : String(value)}
                    </dd>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">No schema information available</p>
            )}
          </CardContent>
          <CardFooter>
            <Button variant="secondary" size="sm" asChild>
              <Link to={`/datasets/${dataset.id}/preview`}>
                <Eye className="h-4 w-4 mr-1" />
                View Data Preview
              </Link>
            </Button>
          </CardFooter>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Usage</CardTitle>
          <CardDescription>How this dataset is being used in evaluations</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            This information will be available once evaluations are implemented.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}