import React, { useState } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { ChevronLeft, FileText, Download, AlertTriangle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { getDataset, getDatasetPreview } from '@/api/datasets';
import { Input } from '@/components/ui/input';

export function DatasetPreview() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [previewLimit, setPreviewLimit] = useState(50);

  const { data: dataset } = useQuery({
    queryKey: ['dataset', id],
    queryFn: () => getDataset(id!),
    enabled: !!id,
  });

  const { data: preview, isLoading, error } = useQuery({
    queryKey: ['dataset-preview', id, previewLimit],
    queryFn: () => getDatasetPreview(id!, { limit: previewLimit }),
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
          </CardHeader>
          <CardContent>
            <div className="h-80 w-full bg-muted animate-pulse rounded"></div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (error || !preview) {
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={() => navigate(-1)}>
            <ChevronLeft className="h-4 w-4 mr-1" />
            Back
          </Button>
          <h1 className="text-3xl font-bold tracking-tight">Dataset Preview</h1>
        </div>
        <Card className="border-destructive">
          <CardHeader>
            <CardTitle className="flex items-center text-destructive">
              <AlertTriangle className="h-5 w-5 mr-2" />
              Error Loading Preview
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p>Failed to load dataset preview. There might be an issue with the dataset or the server.</p>
          </CardContent>
          <CardFooter>
            <Button onClick={() => navigate(`/datasets/${id}`)}>
              Return to Dataset Details
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
          <h1 className="text-3xl font-bold tracking-tight">Preview: {dataset?.name}</h1>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" asChild>
            <Link to={`/datasets/${id}`}>
              <FileText className="h-4 w-4 mr-1" />
              View Details
            </Link>
          </Button>
          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-1" />
            Download
          </Button>
        </div>
      </div>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <div>
            <CardTitle>Data Preview</CardTitle>
            <CardDescription>
              Showing {Math.min(preview.rows.length, previewLimit)} of {preview.total_rows} rows
            </CardDescription>
          </div>
          <div className="flex items-center space-x-2">
            <label htmlFor="rowLimit" className="text-sm text-muted-foreground">
              Rows to show:
            </label>
            <Input
              id="rowLimit"
              type="number"
              min="1"
              max="500"
              value={previewLimit}
              onChange={(e) => setPreviewLimit(Number(e.target.value))}
              className="w-20"
            />
          </div>
        </CardHeader>
        <CardContent>
          <div className="rounded border overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="bg-muted/50">
                    {preview.headers.map((header, index) => (
                      <th
                        key={index}
                        className="px-4 py-3 text-left text-sm font-medium"
                      >
                        {header}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.rows.length === 0 ? (
                    <tr>
                      <td
                        colSpan={preview.headers.length}
                        className="px-4 py-3 text-center text-muted-foreground"
                      >
                        No data available
                      </td>
                    </tr>
                  ) : (
                    preview.rows.map((row, rowIndex) => (
                      <tr key={rowIndex} className="border-t">
                        {row.map((cell, cellIndex) => (
                          <td
                            key={cellIndex}
                            className="px-4 py-2 text-sm max-w-xs truncate"
                            title={String(cell)}
                          >
                            {cell === null || cell === undefined
                              ? <span className="text-muted-foreground italic">null</span>
                              : String(cell)}
                          </td>
                        ))}
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </CardContent>
        <CardFooter className="text-sm text-muted-foreground">
          {preview.total_rows > preview.rows.length && (
            <p>Note: This is a limited preview. The complete dataset contains {preview.total_rows} rows.</p>
          )}
        </CardFooter>
      </Card>
    </div>
  );
}